<?php
declare(strict_types=1);

/**
 * Fetches an English Wikipedia article and maps it to the model's input fields.
 * Keep in sync with wiki.py (the scripting interface), so both give identical predictions:
 *   title    = article title
 *   keywords = short description + non-hidden categories (one per line)
 *   abstract = plain-text lead section, or the whole article when $scope === 'full'
 */
final class Wikipedia
{
    private const API = 'https://en.wikipedia.org/w/api.php';

    public function __construct(private string $userAgent, private int $timeout = 10)
    {
    }

    /** Article title from a title or an en.wikipedia.org URL. */
    public static function parseTitle(string $input): string
    {
        $input = trim($input);
        if (preg_match('~^(?:https?://)?([a-z0-9-]+)(?:\.m)?\.wikipedia\.org/(?:wiki/([^?#]+)|w/index\.php\?(?:.*&)?title=([^&#]+))~i', $input, $m)) {
            if (strtolower($m[1]) !== 'en') {
                throw new InvalidArgumentException('Only English Wikipedia articles are supported (the model is trained on English text).');
            }
            $input = rawurldecode(($m[2] ?? '') !== '' ? $m[2] : $m[3]);
        }
        $title = trim(str_replace('_', ' ', $input));
        if ($title === '') {
            throw new InvalidArgumentException('Enter a Wikipedia article title or URL.');
        }
        return $title;
    }

    /**
     * @return array{title:string,url:string,description:string,categories:list<string>,text:string,redirected_from:?string}
     */
    public function fetch(string $titleOrUrl, string $scope = 'lead'): array
    {
        $title = self::parseTitle($titleOrUrl);
        $params = [
            'action' => 'query', 'format' => 'json', 'formatversion' => '2', 'redirects' => '1',
            'prop' => 'extracts|categories|info|pageprops|description',
            'explaintext' => '1', 'clshow' => '!hidden', 'cllimit' => 'max',
            'inprop' => 'url', 'ppprop' => 'disambiguation', 'titles' => $title,
        ];
        if ($scope !== 'full') {
            $params['exintro'] = '1';
        }
        $data = json_decode($this->get(self::API . '?' . http_build_query($params)), true);
        $page = $data['query']['pages'][0] ?? null;
        if (!is_array($page) || isset($page['invalid'])) {
            throw new RuntimeException("“{$title}” is not a valid Wikipedia title.");
        }
        if (isset($page['missing'])) {
            throw new RuntimeException("No English Wikipedia article is titled “{$title}”.");
        }
        if (isset($page['pageprops']['disambiguation'])) {
            throw new RuntimeException("“{$page['title']}” is a disambiguation page. Choose a specific article.");
        }
        $redirect = $data['query']['redirects'][0]['from'] ?? null;
        return [
            'title' => $page['title'],
            'url' => $page['fullurl'] ?? 'https://en.wikipedia.org/wiki/' . rawurlencode(str_replace(' ', '_', $page['title'])),
            'description' => (string)($page['description'] ?? ''),
            'categories' => array_map(fn($c) => preg_replace('/^Category:/', '', $c['title']), $page['categories'] ?? []),
            'text' => trim((string)($page['extract'] ?? '')),
            'redirected_from' => $redirect,
        ];
    }

    /** Model input fields for a fetched article. */
    public static function toFields(array $article): array
    {
        $keywords = array_filter(array_merge([$article['description']], $article['categories']), 'strlen');
        return [
            'title' => $article['title'],
            'keywords' => implode("\n", $keywords),
            'abstract' => $article['text'],
        ];
    }

    private function get(string $url): string
    {
        if (function_exists('curl_init')) {
            $ch = curl_init($url);
            curl_setopt_array($ch, [
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_FOLLOWLOCATION => true,
                CURLOPT_TIMEOUT => $this->timeout,
                CURLOPT_USERAGENT => $this->userAgent,
                CURLOPT_ENCODING => '',
            ]);
            $body = curl_exec($ch);
            $status = curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
            $err = curl_error($ch);
            curl_close($ch);
        } else {
            $ctx = stream_context_create(['http' => [
                'timeout' => $this->timeout, 'user_agent' => $this->userAgent, 'ignore_errors' => true,
            ]]);
            $body = @file_get_contents($url, false, $ctx);
            $status = preg_match('~^HTTP/\S+ (\d+)~', $http_response_header[0] ?? '', $m) ? (int)$m[1] : 0;
            $err = $body === false ? 'request failed' : '';
        }
        if ($body === false || $status !== 200) {
            throw new RuntimeException('Could not reach Wikipedia' . ($err ? " ($err)" : " (HTTP $status)") . '. Try again shortly.');
        }
        return $body;
    }
}
