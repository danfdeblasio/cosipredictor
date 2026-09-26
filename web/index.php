<?php
declare(strict_types=1);

require __DIR__ . '/lib/app.php';

$input = array_fill_keys(COSI_FIELDS, '');
$ranked = null;
$error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $input = cosi_input($_POST);
    if (implode('', $input) === '') {
        $error = 'Enter a title, keywords or an abstract (at least one).';
    } else {
        try {
            $ranked = cosi_rank($input);
        } catch (Throwable $e) {
            error_log('COSI predictor: ' . $e->getMessage());
            $error = 'Sorry, the prediction could not be computed. Please try again later.';
        }
    }
}

page_header('index', 'Paste a submission’s title, keywords or abstract to see which ISMB Community of Special Interest (COSI) it fits best.');
?>
<main class="wrap layout">
  <section class="card form-card" aria-labelledby="form-heading">
    <h2 id="form-heading">Your submission</h2>
    <p class="hint">All fields are optional, but fill in at least one. More text gives better suggestions.</p>
    <form method="post" action="./#results" id="predict-form" novalidate>
      <label for="title">Title</label>
      <input type="text" id="title" name="title" value="<?= h($input['title']) ?>" autocomplete="off">

      <label for="keywords">Keywords <span class="sub">one per line or comma-separated</span></label>
      <textarea id="keywords" name="keywords" rows="3"><?= h($input['keywords']) ?></textarea>

      <label for="abstract">Abstract</label>
      <textarea id="abstract" name="abstract" rows="12"><?= h($input['abstract']) ?></textarea>

      <p class="form-error" id="form-error" role="alert"<?= $error ? '' : ' hidden' ?>><?= h($error ?? '') ?></p>

      <div class="actions">
        <button type="submit">Suggest COSIs</button>
        <a class="reset" href="./">Clear</a>
      </div>
    </form>
  </section>

  <section class="card results-card" id="results" aria-labelledby="results-heading" aria-live="polite">
<?php $ranked ? render_results($ranked) : render_cosi_list('Your ranked COSIs will appear here, each with a predicted match percentage.'); ?>
  </section>
</main>

<script>
document.getElementById('predict-form').addEventListener('submit', function (e) {
  var filled = ['title', 'keywords', 'abstract'].some(function (id) {
    return document.getElementById(id).value.trim() !== '';
  });
  var err = document.getElementById('form-error');
  if (!filled) {
    e.preventDefault();
    err.textContent = 'Enter a title, keywords or an abstract (at least one).';
    err.hidden = false;
    document.getElementById('title').focus();
  }
});
document.getElementById('predict-form').addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) this.requestSubmit();
});
</script>
<?php page_footer(); ?>
