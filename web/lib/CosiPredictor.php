<?php
declare(strict_types=1);

/**
 * Scores text against the exported TF-IDF + logistic regression model
 * (see export_model.py). Mirrors scikit-learn's TfidfVectorizer and
 * LogisticRegression.predict_proba so results match the Python model.
 */
final class CosiPredictor
{
    private PDO $db;
    private array $classes;
    private array $intercept;
    private array $fields;
    private array $stopWords;
    private array $accentMap;
    private string $tokenPattern;

    public function __construct(string $sqlitePath)
    {
        if (!is_readable($sqlitePath)) {
            throw new RuntimeException("Model file not found: $sqlitePath");
        }
        $this->db = new PDO('sqlite:' . $sqlitePath, null, null, [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::SQLITE_ATTR_OPEN_FLAGS => PDO::SQLITE_OPEN_READONLY,
        ]);
        $meta = [];
        foreach ($this->db->query('SELECT key, value FROM meta') as $row) {
            $meta[$row['key']] = json_decode($row['value'], true, 512, JSON_THROW_ON_ERROR);
        }
        $this->classes = $meta['classes'];
        $this->intercept = $meta['intercept'];
        $this->fields = $meta['fields'];
        $this->stopWords = array_flip($meta['stop_words']);
        $this->accentMap = $meta['accent_map'];
        // Python's (?u)\b\w\w+\b; PCRE's /u flag makes \w and \b Unicode-aware.
        $this->tokenPattern = '/' . str_replace('(?u)', '', $meta['token_pattern']) . '/u';
    }

    public function classes(): array
    {
        return $this->classes;
    }

    /**
     * @param array<string,string> $text  keys: title, keywords, abstract
     * @return array<string,float>         COSI => probability, summing to 1
     */
    public function predict(array $text): array
    {
        $scores = $this->intercept;
        foreach ($this->fields as $field) {
            $counts = $this->termCounts((string)($text[$field['column']] ?? ''), $field);
            if (!$counts) {
                continue;
            }
            $weights = [];
            $coefs = [];
            foreach ($this->lookup($field['name'], array_keys($counts)) as $term => [$idf, $coef]) {
                $tf = $counts[$term];
                if ($field['sublinear_tf']) {
                    $tf = 1.0 + log($tf);
                }
                $weights[$term] = $tf * $idf;
                $coefs[$term] = $coef;
            }
            $norm = sqrt(array_sum(array_map(fn($w) => $w * $w, $weights)));
            if ($norm == 0.0) {
                continue;
            }
            foreach ($weights as $term => $w) {
                $w /= $norm;
                foreach ($coefs[$term] as $k => $c) {
                    $scores[$k] += $w * $c;
                }
            }
        }
        return array_combine($this->classes, self::softmax($scores));
    }

    /** Word n-gram counts, following sklearn's analyzer='word' pipeline. */
    private function termCounts(string $doc, array $field): array
    {
        if ($field['lowercase']) {
            $doc = mb_strtolower($doc, 'UTF-8');
        }
        if ($field['strip_accents'] === 'unicode') {
            $doc = $this->stripAccents($doc);
        }
        preg_match_all($this->tokenPattern, $doc, $m);
        $tokens = array_values(array_filter($m[0], fn($t) => !isset($this->stopWords[$t])));

        [$minN, $maxN] = $field['ngram_range'];
        $counts = [];
        $n = count($tokens);
        for ($size = $minN; $size <= $maxN; $size++) {
            for ($i = 0; $i + $size <= $n; $i++) {
                $gram = implode(' ', array_slice($tokens, $i, $size));
                $counts[$gram] = ($counts[$gram] ?? 0) + 1;
            }
        }
        return $counts;
    }

    /** NFKD + drop combining marks, via a table exported from Python's unicodedata (no intl needed). */
    private function stripAccents(string $doc): string
    {
        return strtr($doc, $this->accentMap);
    }

    /** @return array<string,array{0:float,1:float[]}> term => [idf, coefficients per class] */
    private function lookup(string $field, array $terms): array
    {
        $found = [];
        foreach (array_chunk($terms, 500) as $chunk) {
            $placeholders = implode(',', array_fill(0, count($chunk), '?'));
            $stmt = $this->db->prepare(
                "SELECT term, idf, coef FROM terms WHERE field = ? AND term IN ($placeholders)"
            );
            $stmt->execute(array_merge([$field], $chunk));
            foreach ($stmt->fetchAll(PDO::FETCH_NUM) as [$term, $idf, $blob]) {
                $found[$term] = [(float)$idf, array_values(unpack('g*', $blob))];  // little-endian float32
            }
        }
        return $found;
    }

    private static function softmax(array $z): array
    {
        $max = max($z);
        $exp = array_map(fn($v) => exp($v - $max), $z);
        $sum = array_sum($exp);
        return array_map(fn($v) => $v / $sum, $exp);
    }
}
