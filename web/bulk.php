<?php
declare(strict_types=1);

require __DIR__ . '/lib/app.php';

page_header('bulk', 'Upload a CSV of abstracts or a list of Wikipedia articles and download it back with a probability column for every COSI.');
?>
<main class="wrap bulk-layout">
  <section class="card" aria-labelledby="bulk-heading">
    <h2 id="bulk-heading">Bulk prediction</h2>
    <p class="hint">Your file is read in your browser and sent to the model in small batches. Nothing is stored on the server.</p>

    <fieldset class="modes">
      <legend>What’s in your file?</legend>
      <label class="mode">
        <input type="radio" name="mode" value="abstracts" checked>
        <span class="mode-body">
          <strong>Abstracts</strong>
          <span>Columns named <code>title</code>, <code>keywords</code> and <code>abstract</code> (any order; other columns are kept). With no header row, the first three columns are read in that order.</span>
          <a href="#" class="sample" data-sample="abstracts">Download a sample file</a>
        </span>
      </label>
      <label class="mode">
        <input type="radio" name="mode" value="wikipedia">
        <span class="mode-body">
          <strong>Wikipedia articles</strong>
          <span>One English Wikipedia URL or article title per row: the first column, or a column named <code>url</code>, <code>article</code> or <code>title</code>.</span>
          <a href="#" class="sample" data-sample="wikipedia">Download a sample file</a>
        </span>
      </label>
    </fieldset>

    <fieldset class="choice" id="scope-choice" hidden>
      <legend>Wikipedia text to analyze</legend>
      <label><input type="radio" name="scope" value="lead" checked> Lead section <span class="sub">closest to an abstract (recommended)</span></label>
      <label><input type="radio" name="scope" value="full"> Whole article</label>
    </fieldset>

    <label class="dropzone" id="dropzone">
      <input type="file" id="file" accept=".csv,.tsv,.txt,text/csv,text/plain">
      <span class="dz-title">Choose a CSV file</span>
      <span class="dz-sub">or drag it here</span>
    </label>

    <div class="file-info" id="file-info" hidden></div>
    <p class="form-error" id="bulk-error" role="alert" hidden></p>

    <div class="actions">
      <button type="button" id="run" disabled>Predict COSIs</button>
      <button type="button" id="cancel" class="secondary" hidden>Cancel</button>
    </div>

    <div class="progress" id="progress" hidden>
      <div class="progress-track"><div class="progress-bar" id="progress-bar"></div></div>
      <p class="progress-text" id="progress-text" aria-live="polite"></p>
    </div>

    <div class="done" id="done" hidden>
      <button type="button" id="download">Download results CSV</button>
      <p class="sub" id="done-text"></p>
    </div>
  </section>

  <section class="card" id="preview-card" aria-labelledby="preview-heading" hidden>
    <h2 id="preview-heading">Preview</h2>
    <p class="hint">First rows of the output. The downloaded CSV has every input column followed by <code>status</code>, <code>predicted_cosi</code> and a <code>p_&lt;COSI&gt;</code> probability column for each COSI.</p>
    <div class="table-wrap"><table class="preview" id="preview"></table></div>
  </section>

  <section class="card format-card" aria-labelledby="format-heading">
    <h2 id="format-heading">Output columns</h2>
    <ul class="format-list">
      <li><code>wiki_title</code>, <code>wiki_url</code>: the resolved article (Wikipedia files only)</li>
      <li><code>status</code>: <code>ok</code>, or why the row couldn’t be scored</li>
      <li><code>predicted_cosi</code>: the most likely COSI</li>
      <li><code>p_3DSIG</code> … <code>p_VarI</code>: probability for each of the <?= count(cosi_display_order()) ?> COSIs (each row sums to 1)</li>
    </ul>
  </section>
</main>
<script>
window.COSI_BULK = {
  endpoint: 'batch_api.php',
  chunk: { abstracts: 25, wikipedia: 5 },
  maxRows: { abstracts: 10000, wikipedia: 1000 }
};
</script>
<script src="assets/bulk.js?v=1"></script>
<?php page_footer(); ?>
