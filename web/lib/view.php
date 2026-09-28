<?php
declare(strict_types=1);

// Shared page chrome and the ranked-results chart.

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function page_header(string $active, string $intro): void
{
    $tabs = ['index' => ['./', 'Abstract'], 'wikipedia' => ['wikipedia.php', 'Wikipedia article'], 'bulk' => ['bulk.php', 'Bulk CSV'], 'terms' => ['terms.php', 'COSI vocabulary']];
    ?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>COSI Predictor</title>
<meta name="description" content="Suggests which ISMB Community of Special Interest (COSI) best fits a submission or Wikipedia article.">
<link rel="stylesheet" href="assets/style.css?v=6">
</head>
<body>
<header class="site-header">
  <div class="wrap">
    <h1>COSI Predictor</h1>
    <p><?= h($intro) ?></p>
    <nav class="tabs" aria-label="Input type">
<?php foreach ($tabs as $key => [$href, $label]): ?>
      <a href="<?= h($href) ?>"<?= $key === $active ? ' aria-current="page"' : '' ?>><?= h($label) ?></a>
<?php endforeach; ?>
    </nav>
  </div>
</header>
<?php
}

function page_footer(string $extra = ''): void
{
    ?>
<footer class="wrap site-footer">
  <p>These are suggestions from a text classifier (TF-IDF + logistic regression) trained on 9,103 accepted ISMB and ISMB/ECCB COSI submissions from 2019–2026. Tested on ISMB 2026 submissions it hadn’t seen, its top suggestion matched the submission’s COSI 65% of the time, and that COSI was in its top three 91% of the time. Always check the COSI’s scope before choosing.</p>
<?php if ($extra !== ''): ?>
  <p><?= h($extra) ?></p>
<?php endif; ?>
  <p>Submitted text is not stored.</p>
<?php $repo = cosi_config()['repo_url'] ?? ''; ?>
<?php if ($repo !== ''): ?>
  <p class="footer-links">
    <a href="<?= h($repo) ?>">Source code on GitHub</a>
    <span aria-hidden="true">·</span>
    <a href="<?= h($repo) ?>/blob/main/METHODS.md">Methods</a>
  </p>
<?php endif; ?>
</footer>
</body>
</html>
<?php
}

/** Best-match summary plus the bar chart of every COSI. */
function render_results(array $ranked): void
{
    $top = $ranked[0];
    $second = $ranked[1] ?? null;
    $closeCall = $second && ($top['probability'] - $second['probability']) < 0.10;
    ?>
    <h2 id="results-heading" class="eyebrow">Best match</h2>
    <div class="best">
      <div class="best-name">
        <span class="abbr"><?= h($top['cosi']) ?></span>
        <span class="full"><?= h($top['name']) ?></span>
      </div>
      <div class="best-pct"><?= h(cosi_pct($top['probability'])) ?></div>
    </div>
    <p class="verdict">
<?php if ($closeCall): ?>
      Close call with <strong><?= h($second['cosi']) ?></strong> (<?= h(cosi_pct($second['probability'])) ?>). Read both COSIs’ scopes before choosing.
<?php elseif ($top['probability'] >= 0.5): ?>
      A clear favorite. Also worth a look: <?= implode(', ', array_map(fn($r) => '<strong>' . h($r['cosi']) . '</strong>', array_slice($ranked, 1, 2))) ?>.
<?php else: ?>
      No single strong fit. Compare the top three below.
<?php endif; ?>
    </p>

    <div class="chart" role="figure" aria-labelledby="chart-caption">
      <p id="chart-caption" class="chart-caption">Predicted match for every COSI</p>
      <div class="axis" aria-hidden="true">
        <span style="left:0">0%</span><span style="left:25%">25%</span><span style="left:50%">50%</span><span style="left:75%">75%</span><span style="left:100%">100%</span>
      </div>
      <ol class="bars">
<?php foreach ($ranked as $i => $r): ?>
<?php if ($i === 3): ?>
        <li class="divider" aria-hidden="true"><span>Less likely</span></li>
<?php endif; ?>
        <li class="bar-row<?= $i < 3 ? ' top' : '' ?>" title="<?= h($r['cosi'] . ($r['name'] ? ' – ' . $r['name'] : '') . ': ' . cosi_pct($r['probability'])) ?>">
          <span class="label"><span class="abbr"><?= h($r['cosi']) ?></span> <span class="full"><?= h($r['name']) ?></span></span>
          <span class="track"><span class="bar" style="width:<?= round($r['probability'] * 100, 2) ?>%"></span></span>
          <span class="value"><?= h(cosi_pct($r['probability'])) ?></span>
        </li>
<?php endforeach; ?>
      </ol>
    </div>
<?php
}

/** Placeholder listing the COSIs the model chooses among. */
function render_cosi_list(string $what): void
{
    $config = cosi_config();
    $list = $config['active_cosis'] ?? array_keys($config['names']);
    natcasesort($list);
    ?>
    <h2 id="results-heading">Suggested COSIs</h2>
    <p class="hint"><?= h($what) ?> The model chooses among these <?= count($list) ?> COSIs:</p>
    <ul class="cosi-list">
<?php foreach ($list as $c): ?>
      <li><span class="abbr"><?= h($c) ?></span> <span class="full"><?= h($config['names'][$c] ?? '') ?></span></li>
<?php endforeach; ?>
    </ul>
<?php
}
