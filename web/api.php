<?php
declare(strict_types=1);

// JSON API: POST title / keywords / abstract (form-encoded or JSON body).
// Returns every COSI ranked by predicted match probability.
require __DIR__ . '/lib/app.php';

header('Content-Type: application/json; charset=utf-8');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    header('Allow: POST');
    echo json_encode(['error' => 'Use POST with title, keywords and/or abstract.']);
    exit;
}

$body = $_POST;
if (str_starts_with($_SERVER['CONTENT_TYPE'] ?? '', 'application/json')) {
    $body = json_decode((string)file_get_contents('php://input'), true);
    if (!is_array($body)) {
        http_response_code(400);
        echo json_encode(['error' => 'Invalid JSON body.']);
        exit;
    }
}

// Wikipedia mode: {"wikipedia": "<title or URL>", "scope": "lead" | "full"}
if (is_string($body['wikipedia'] ?? null) && trim($body['wikipedia']) !== '') {
    try {
        $article = wikipedia_client()->fetch($body['wikipedia'], ($body['scope'] ?? '') === 'full' ? 'full' : 'lead');
        $fields = Wikipedia::toFields($article);
        echo json_encode([
            'article' => ['title' => $article['title'], 'url' => $article['url'],
                          'description' => $article['description'], 'categories' => $article['categories']],
            'predictions' => cosi_rank($fields),
        ], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    } catch (InvalidArgumentException | RuntimeException $e) {
        http_response_code(422);
        echo json_encode(['error' => $e->getMessage()], JSON_UNESCAPED_UNICODE);
    } catch (Throwable $e) {
        error_log('COSI predictor (wikipedia): ' . $e->getMessage());
        http_response_code(500);
        echo json_encode(['error' => 'Prediction failed.']);
    }
    exit;
}

$input = cosi_input($body);
if (implode('', $input) === '') {
    http_response_code(400);
    echo json_encode(['error' => 'Provide title, keywords and/or abstract, or a wikipedia article.']);
    exit;
}

try {
    echo json_encode(['predictions' => cosi_rank($input)], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
} catch (Throwable $e) {
    error_log('COSI predictor: ' . $e->getMessage());
    http_response_code(500);
    echo json_encode(['error' => 'Prediction failed.']);
}
