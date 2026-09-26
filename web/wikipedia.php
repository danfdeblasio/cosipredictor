<?php
declare(strict_types=1);

require __DIR__ . '/lib/app.php';

$query = is_string($_GET['article'] ?? null) ? trim($_GET['article']) : '';
$scope = ($_GET['scope'] ?? '') === 'full' ? 'full' : 'lead';
$article = null;
$ranked = null;
$error = null;

if ($query !== '') {
    try {
        $article = wikipedia_client()->fetch($query, $scope);
        $ranked = cosi_rank(Wikipedia::toFields($article));
    } catch (InvalidArgumentException | RuntimeException $e) {
        $error = $e->getMessage();
    } catch (Throwable $e) {
        error_log('COSI predictor (wikipedia): ' . $e->getMessage());
        $error = 'Sorry, the prediction could not be computed. Please try again later.';
    }
}

page_header('wikipedia', 'Look up an English Wikipedia article to see which ISMB Community of Special Interest (COSI) its topic fits best.');
?>
<main class="wrap layout">
  <div class="stack">
    <section class="card form-card" aria-labelledby="form-heading">
      <h2 id="form-heading">Wikipedia article</h2>
      <p class="hint">Enter an article title or paste its URL.</p>
      <form method="get" action="wikipedia.php#results" id="wiki-form">
        <label for="article">Article</label>
        <input type="text" id="article" name="article" value="<?= h($query) ?>" placeholder="e.g. BLAST (biotechnology)" autocomplete="off" required>

        <fieldset class="choice">
          <legend>Text to analyze</legend>
          <label><input type="radio" name="scope" value="lead"<?= $scope === 'lead' ? ' checked' : '' ?>> Lead section <span class="sub">closest to an abstract (recommended)</span></label>
          <label><input type="radio" name="scope" value="full"<?= $scope === 'full' ? ' checked' : '' ?>> Whole article</label>
        </fieldset>

        <p class="form-error" role="alert"<?= $error ? '' : ' hidden' ?>><?= h($error ?? '') ?></p>

        <div class="actions">
          <button type="submit">Suggest COSIs</button>
          <a class="reset" href="wikipedia.php">Clear</a>
        </div>
      </form>
    </section>

<?php if ($article): ?>
    <section class="card article-card" aria-labelledby="article-heading">
      <p class="eyebrow">Analyzed article</p>
      <h2 id="article-heading"><a href="<?= h($article['url']) ?>" target="_blank" rel="noopener"><?= h($article['title']) ?></a></h2>
<?php if ($article['redirected_from']): ?>
      <p class="sub">Redirected from “<?= h($article['redirected_from']) ?>”</p>
<?php endif; ?>
<?php if ($article['description'] !== ''): ?>
      <p class="description"><?= h($article['description']) ?></p>
<?php endif; ?>
<?php if ($article['categories']): ?>
      <p class="field-label">Categories <span class="sub">used as keywords</span></p>
      <ul class="chips">
<?php foreach ($article['categories'] as $c): ?>
        <li><?= h($c) ?></li>
<?php endforeach; ?>
      </ul>
<?php endif; ?>
      <details class="extract">
        <summary><?= $scope === 'full' ? 'Article text' : 'Lead section' ?> used as the abstract <span class="sub">(<?= number_format(str_word_count($article['text'])) ?> words)</span></summary>
        <div class="extract-text"><?= nl2br(h($article['text'])) ?></div>
      </details>
    </section>
<?php endif; ?>
  </div>

  <section class="card results-card" id="results" aria-labelledby="results-heading" aria-live="polite">
<?php $ranked ? render_results($ranked) : render_cosi_list('Look up an article to see its ranked COSIs, each with a predicted match percentage.'); ?>
  </section>
</main>
<?php page_footer('Article text is fetched live from the Wikipedia API. Wikipedia articles are written differently from conference abstracts, so treat these percentages as a rough guide.'); ?>
