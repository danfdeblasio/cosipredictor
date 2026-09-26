<?php
declare(strict_types=1);

// Batch scoring for bulk.php. The browser parses the uploaded CSV and sends rows in small chunks,
// so a large file never hits PHP's time limit and nothing is stored on the server.
//
// POST JSON:
//   {"mode": "abstracts", "rows": [{"title": "...", "keywords": "...", "abstract": "..."}, ...]}
//   {"mode": "wikipedia", "scope": "lead"|"full", "rows": ["<title or URL>", ...]}
// Returns {"cosis": [...], "results": [...]}, one result per row, in order:
//   abstracts: {"probabilities": {COSI: p, ...}} or {"error": "..."} for rows with no text
//   wikipedia: {"title", "url", "probabilities"} or {"error": "..."}
require __DIR__ . '/lib/app.php';

const MAX_ABSTRACT_ROWS = 100;
const MAX_WIKIPEDIA_ROWS = 10;

header('Content-Type: application/json; charset=utf-8');

function fail(int $status, string $message): never
{
    http_response_code($status);
    echo json_encode(['error' => $message]);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Allow: POST');
    fail(405, 'Use POST.');
}
$body = json_decode((string)file_get_contents('php://input'), true);
if (!is_array($body) || !is_array($body['rows'] ?? null)) {
    fail(400, 'Expected a JSON body with "mode" and "rows".');
}
$mode = $body['mode'] ?? '';
$rows = array_values($body['rows']);

function rounded(array $probs): array
{
    return array_map(fn($p) => round($p, 6), $probs);
}

try {
    $results = [];
    if ($mode === 'abstracts') {
        if (count($rows) > MAX_ABSTRACT_ROWS) {
            fail(413, 'At most ' . MAX_ABSTRACT_ROWS . ' rows per request.');
        }
        foreach ($rows as $row) {
            $input = cosi_input(is_array($row) ? $row : []);
            $results[] = implode('', $input) === ''
                ? ['error' => 'no title, keywords or abstract']
                : ['probabilities' => rounded(cosi_probabilities($input))];
        }
    } elseif ($mode === 'wikipedia') {
        if (count($rows) > MAX_WIKIPEDIA_ROWS) {
            fail(413, 'At most ' . MAX_WIKIPEDIA_ROWS . ' articles per request.');
        }
        $scope = ($body['scope'] ?? '') === 'full' ? 'full' : 'lead';
        $wiki = wikipedia_client();
        foreach ($rows as $query) {
            try {
                $article = $wiki->fetch(is_string($query) ? $query : '', $scope);
                $results[] = [
                    'title' => $article['title'],
                    'url' => $article['url'],
                    'probabilities' => rounded(cosi_probabilities(Wikipedia::toFields($article))),
                ];
            } catch (InvalidArgumentException | RuntimeException $e) {
                $results[] = ['error' => $e->getMessage()];
            }
        }
    } else {
        fail(400, 'mode must be "abstracts" or "wikipedia".');
    }
    echo json_encode(['cosis' => cosi_display_order(), 'results' => $results],
                     JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
} catch (Throwable $e) {
    error_log('COSI predictor (batch): ' . $e->getMessage());
    fail(500, 'Prediction failed.');
}
