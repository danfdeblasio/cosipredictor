<?php
declare(strict_types=1);

require_once __DIR__ . '/CosiPredictor.php';
require_once __DIR__ . '/Wikipedia.php';
require_once __DIR__ . '/view.php';

const COSI_FIELDS = ['title', 'keywords', 'abstract'];

function cosi_config(): array
{
    static $config = null;
    return $config ??= require __DIR__ . '/../config.php';
}

/** Trimmed, length-limited input fields from a request array. */
function cosi_input(array $source): array
{
    $max = cosi_config()['max_chars'];
    $out = [];
    foreach (COSI_FIELDS as $f) {
        $v = is_string($source[$f] ?? null) ? trim($source[$f]) : '';
        $out[$f] = mb_substr($v, 0, $max, 'UTF-8');
    }
    return $out;
}

function cosi_predictor(): CosiPredictor
{
    static $predictor = null;
    return $predictor ??= new CosiPredictor(cosi_config()['model_path']);
}

/**
 * Probabilities for the configured COSIs (renormalized when active_cosis is set).
 * @return array<string,float> COSI => probability, in model order
 */
function cosi_probabilities(array $input): array
{
    $config = cosi_config();
    $probs = cosi_predictor()->predict($input);
    if ($config['active_cosis'] !== null) {
        $probs = array_intersect_key($probs, array_flip($config['active_cosis']));
        $total = array_sum($probs);
        $probs = array_map(fn($p) => $p / $total, $probs);
    }
    return $probs;
}

/** COSIs shown to users, in case-insensitive alphabetical order. */
function cosi_display_order(): array
{
    $config = cosi_config();
    $list = $config['active_cosis'] ?? cosi_predictor()->classes();
    natcasesort($list);
    return array_values($list);
}

/**
 * Ranked predictions for the configured COSIs.
 * @return list<array{cosi:string,name:string,probability:float}>
 */
function cosi_rank(array $input): array
{
    $config = cosi_config();
    $probs = cosi_probabilities($input);
    arsort($probs);
    $ranked = [];
    foreach ($probs as $cosi => $p) {
        $ranked[] = ['cosi' => (string)$cosi, 'name' => $config['names'][$cosi] ?? '', 'probability' => $p];
    }
    return $ranked;
}

function wikipedia_client(): Wikipedia
{
    return new Wikipedia(cosi_config()['wikipedia_user_agent']);
}

function cosi_pct(float $p): string
{
    $pct = $p * 100;
    if ($pct < 0.1) {
        return '<0.1%';
    }
    return ($pct < 10 ? number_format($pct, 1) : (string)round($pct)) . '%';
}
