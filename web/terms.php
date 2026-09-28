<?php
declare(strict_types=1);

require __DIR__ . '/lib/app.php';

$config = cosi_config();
$data = null;
$path = __DIR__ . '/data/top_terms.json';
if (is_readable($path)) {
    $data = json_decode((string)file_get_contents($path), true);
}

$cosis = [];
$maxScore = 0.0;
if ($data) {
    foreach (cosi_display_order() as $c) {
        if (isset($data['cosis'][$c])) {
            $cosis[$c] = $data['cosis'][$c];
            foreach ($cosis[$c]['terms'] as $t) {
                $maxScore = max($maxScore, $t['score']);
            }
        }
    }
}
// Shared scale for every chart, rounded up to the next 0.05.
$scaleMax = max(0.05, ceil($maxScore / 0.05) * 0.05);
$gridStep = 0.05 / $scaleMax * 100;

function anchor(string $cosi): string
{
    return 'cosi-' . preg_replace('/[^A-Za-z0-9-]/', '', $cosi);
}

page_header('terms', 'The ten word stems that characterize each COSI’s accepted submissions, ranked by TF-IDF score.');
?>
<main class="wrap">
<?php if (!$cosis): ?>
  <section class="card"><p class="form-error">The vocabulary data is not available.</p></section>
<?php else: ?>
  <section class="card vocab-intro" aria-labelledby="vocab-heading">
    <h2 id="vocab-heading">How to read this page</h2>
    <p class="hint">
      Each bar is a stem’s average TF-IDF weight across that COSI’s training submissions (title, keywords and abstract combined; <?= number_format($data['submissions']) ?> submissions in total).
      A stem scores highly when it appears in many of the COSI’s submissions and is uncommon across ISMB overall.
      Stems group word forms (e.g. <em>sequence</em>, <em>sequences</em>, <em>sequencing</em>) and are shown as their most frequent form; hover over a stem to see its forms.
      All bars share one scale (0–<?= h(number_format($scaleMax, 2)) ?>), so lengths can be compared across COSIs.
    </p>
    <nav class="vocab-jump" aria-label="Jump to COSI">
<?php foreach (array_keys($cosis) as $c): ?>
      <a href="#<?= h(anchor($c)) ?>"><?= h($c) ?></a>
<?php endforeach; ?>
    </nav>
  </section>

  <div class="vocab-grid">
<?php foreach ($cosis as $c => $info): ?>
    <section class="card vocab-card" id="<?= h(anchor($c)) ?>" aria-labelledby="<?= h(anchor($c)) ?>-h">
      <h3 id="<?= h(anchor($c)) ?>-h"><span class="abbr"><?= h($c) ?></span></h3>
      <p class="vocab-name"><?= h($config['names'][$c] ?? '') ?></p>
      <p class="vocab-n"><?= number_format($info['submissions']) ?> submissions</p>
      <ol class="vocab-bars" aria-label="Top stems for <?= h($c) ?>">
<?php foreach ($info['terms'] as $t): ?>
        <li title="<?= h('Stem “' . $t['stem'] . '”: ' . implode(', ', $t['forms']) . ' · TF-IDF ' . number_format($t['score'], 3)) ?>">
          <span class="vocab-word"><?= h($t['word']) ?></span>
          <span class="vocab-track" style="background-size: <?= round($gridStep, 3) ?>% 100%"><span class="vocab-bar" style="width: <?= round($t['score'] / $scaleMax * 100, 2) ?>%"></span></span>
          <span class="vocab-value"><?= h(number_format($t['score'], 3)) ?></span>
        </li>
<?php endforeach; ?>
      </ol>
    </section>
<?php endforeach; ?>
  </div>
<?php endif; ?>
</main>
<?php page_footer('Stems are computed from the same training data as the classifier. Generated ' . ($data['generated'] ?? '') . '.'); ?>
