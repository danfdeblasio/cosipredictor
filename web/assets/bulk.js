(function () {
  'use strict';

  var cfg = window.COSI_BULK;
  var $ = function (id) { return document.getElementById(id); };
  var state = { rows: null, header: null, hasHeader: false, cols: null, fileName: '', output: null, cancelled: false };

  // ---------- CSV parsing / writing ----------

  function detectDelimiter(text) {
    var line = text.slice(0, 10000).split(/\r?\n/)[0];
    var best = ',', bestCount = 0;
    [',', '\t', ';'].forEach(function (d) {
      var n = line.split(d).length - 1;
      if (n > bestCount) { best = d; bestCount = n; }
    });
    return best;
  }

  // RFC 4180: quoted fields may contain delimiters, doubled quotes and newlines.
  function parseCSV(text) {
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    var delim = detectDelimiter(text);
    var rows = [], row = [], field = '', inQuotes = false, c;
    for (var i = 0; i < text.length; i++) {
      c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; } else { inQuotes = false; }
        } else {
          field += c;
        }
      } else if (c === '"' && field === '') {
        inQuotes = true;
      } else if (c === delim) {
        row.push(field); field = '';
      } else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field); rows.push(row); row = []; field = '';
      } else {
        field += c;
      }
    }
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    return rows.filter(function (r) { return r.some(function (v) { return v.trim() !== ''; }); });
  }

  function csvCell(v) {
    v = v == null ? '' : String(v);
    return /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }

  function toCSV(rows) {
    return '﻿' + rows.map(function (r) { return r.map(csvCell).join(','); }).join('\r\n') + '\r\n';
  }

  function download(name, text) {
    var blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  // ---------- Column detection ----------

  function norm(s) { return String(s).toLowerCase().replace(/[^a-z0-9]/g, ''); }

  function detectColumns(rows, mode) {
    var first = rows[0].map(norm);
    if (mode === 'abstracts') {
      var find = function (names) {
        for (var i = 0; i < first.length; i++) if (names.indexOf(first[i]) >= 0) return i;
        return -1;
      };
      var cols = { title: find(['title']), keywords: find(['keywords', 'keyword']), abstract: find(['abstract']) };
      if (cols.title >= 0 || cols.keywords >= 0 || cols.abstract >= 0) return { hasHeader: true, cols: cols };
      return { hasHeader: false, cols: { title: 0, keywords: rows[0].length > 1 ? 1 : -1, abstract: rows[0].length > 2 ? 2 : -1 } };
    }
    var names = ['url', 'urls', 'article', 'articles', 'wikipedia', 'wikipediaurl', 'wikipediaarticle',
                 'page', 'title', 'link', 'name'];
    for (var i = 0; i < first.length; i++) {
      if (names.indexOf(first[i]) >= 0 && !/wikipedia\.org/i.test(rows[0][i])) {
        return { hasHeader: true, cols: { article: i } };
      }
    }
    return { hasHeader: false, cols: { article: 0 } };
  }

  function outputHeader(mode) {
    var width = Math.max.apply(null, state.rows.map(function (r) { return r.length; }));
    var header = [];
    for (var i = 0; i < width; i++) {
      if (state.hasHeader && state.header[i] !== undefined && state.header[i] !== '') header.push(state.header[i]);
      else if (!state.hasHeader && mode === 'abstracts' && i < 3) header.push(['title', 'keywords', 'abstract'][i]);
      else if (!state.hasHeader && mode === 'wikipedia' && i === 0) header.push('article');
      else header.push('column_' + (i + 1));
    }
    return { cols: header, width: width };
  }

  // ---------- UI ----------

  function mode() { return document.querySelector('input[name="mode"]:checked').value; }
  function scope() { return document.querySelector('input[name="scope"]:checked').value; }

  function showError(msg) {
    $('bulk-error').textContent = msg;
    $('bulk-error').hidden = !msg;
  }

  function colLabel(i) {
    if (i < 0) return '<em>not found</em>';
    var name = state.hasHeader ? state.header[i] : 'column ' + (i + 1);
    return '<code>' + escapeHTML(name || 'column ' + (i + 1)) + '</code>';
  }

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function describe() {
    var m = mode();
    if (!state.allRows) return;
    var det = detectColumns(state.allRows, m);
    state.hasHeader = det.hasHeader;
    state.cols = det.cols;
    state.header = det.hasHeader ? state.allRows[0] : null;
    state.rows = det.hasHeader ? state.allRows.slice(1) : state.allRows.slice();
    state.output = null;
    $('done').hidden = true;
    $('preview-card').hidden = true;

    var n = state.rows.length;
    var info = '<strong>' + escapeHTML(state.fileName) + '</strong>: ' + n.toLocaleString() + ' row' + (n === 1 ? '' : 's') +
      (state.hasHeader ? ' (plus a header row)' : ' (no header row found)') + '<br>';
    if (m === 'abstracts') {
      info += 'Title: ' + colLabel(state.cols.title) + ' · Keywords: ' + colLabel(state.cols.keywords) +
              ' · Abstract: ' + colLabel(state.cols.abstract);
    } else {
      info += 'Articles: ' + colLabel(state.cols.article) +
              (n ? ' (e.g. “' + escapeHTML(state.rows[0][state.cols.article] || '') + '”)' : '');
    }
    $('file-info').innerHTML = info;
    $('file-info').hidden = false;

    var err = '';
    if (!n) err = 'The file has no data rows.';
    else if (n > cfg.maxRows[m]) err = 'This file has ' + n.toLocaleString() + ' rows; the limit is ' + cfg.maxRows[m].toLocaleString() + ' for ' + (m === 'abstracts' ? 'abstracts' : 'Wikipedia articles') + '.';
    showError(err);
    $('run').disabled = !!err;
  }

  function readFile(file) {
    if (!file) return;
    state.fileName = file.name;
    var reader = new FileReader();
    reader.onload = function () {
      try {
        state.allRows = parseCSV(reader.result);
      } catch (e) {
        showError('Could not read this file as CSV.');
        return;
      }
      if (!state.allRows.length) {
        showError('The file is empty.');
        $('run').disabled = true;
        return;
      }
      describe();
    };
    reader.onerror = function () { showError('Could not read the file.'); };
    reader.readAsText(file);
  }

  function setProgress(done, total, errors) {
    var pct = total ? Math.round(100 * done / total) : 0;
    $('progress-bar').style.width = pct + '%';
    $('progress-text').textContent = done.toLocaleString() + ' of ' + total.toLocaleString() + ' rows (' + pct + '%)' +
      (errors ? ' · ' + errors + ' could not be scored' : '');
  }

  function post(body, attempt) {
    attempt = attempt || 0;
    return fetch(cfg.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok || !data.results) throw new Error(data.error || 'HTTP ' + res.status);
        return data;
      });
    }).catch(function (e) {
      if (attempt >= 2 || state.cancelled) throw e;
      return new Promise(function (r) { setTimeout(r, 1500 * (attempt + 1)); })
        .then(function () { return post(body, attempt + 1); });
    });
  }

  function cell(row, i) { return i >= 0 && row[i] !== undefined ? row[i] : ''; }

  function run() {
    var m = mode(), sc = scope(), rows = state.rows, size = cfg.chunk[m];
    var results = new Array(rows.length), cosis = null, errors = 0, done = 0;
    state.cancelled = false;
    showError('');
    $('run').disabled = true;
    $('cancel').hidden = false;
    $('done').hidden = true;
    $('progress').hidden = false;
    setProgress(0, rows.length, 0);

    var start = 0;
    function next() {
      if (state.cancelled) return finish(true);
      if (start >= rows.length) return finish(false);
      var chunk = rows.slice(start, start + size), offset = start;
      start += size;
      var body = m === 'abstracts'
        ? { mode: m, rows: chunk.map(function (r) {
            return { title: cell(r, state.cols.title), keywords: cell(r, state.cols.keywords), abstract: cell(r, state.cols.abstract) };
          }) }
        : { mode: m, scope: sc, rows: chunk.map(function (r) { return cell(r, state.cols.article); }) };
      return post(body).then(function (data) {
        cosis = cosis || data.cosis;
        data.results.forEach(function (res, j) { results[offset + j] = res; if (res.error) errors++; });
      }, function (e) {
        chunk.forEach(function (_, j) { results[offset + j] = { error: 'request failed (' + e.message + ')' }; });
        errors += chunk.length;
      }).then(function () {
        done += chunk.length;
        setProgress(done, rows.length, errors);
        return next();
      });
    }

    function finish(cancelled) {
      $('cancel').hidden = true;
      $('run').disabled = false;
      if (!cosis) {
        showError(cancelled ? 'Cancelled.' : 'The server could not score any rows. Please try again later.');
        return;
      }
      var head = outputHeader(m);
      var extra = (m === 'wikipedia' ? ['wiki_title', 'wiki_url'] : [])
        .concat(['status', 'predicted_cosi'], cosis.map(function (c) { return 'p_' + c; }));
      var out = [head.cols.concat(extra)];
      var scored = 0;
      rows.forEach(function (r, i) {
        var line = r.slice();
        while (line.length < head.width) line.push('');
        var res = results[i];
        if (!res) {
          res = { error: 'not processed (cancelled)' };
        }
        if (m === 'wikipedia') line.push(res.title || '', res.url || '');
        if (res.probabilities) {
          scored++;
          var best = cosis.reduce(function (a, c) { return res.probabilities[c] > res.probabilities[a] ? c : a; }, cosis[0]);
          line.push('ok', best);
          cosis.forEach(function (c) { line.push(res.probabilities[c].toFixed(4)); });
        } else {
          line.push('error: ' + res.error, '');
          cosis.forEach(function () { line.push(''); });
        }
        out.push(line);
      });
      state.output = out;
      state.outName = state.fileName.replace(/\.[^.]+$/, '') + '_cosi_predictions.csv';
      $('done-text').textContent = scored.toLocaleString() + ' of ' + rows.length.toLocaleString() + ' rows scored' +
        (cancelled ? ' (cancelled early)' : '') + '.';
      $('done').hidden = false;
      renderPreview(out, m);
      download(state.outName, toCSV(out));
    }

    next();
  }

  function renderPreview(out, m) {
    var header = out[0], statusIdx = header.lastIndexOf('status');
    var textIdx = m === 'abstracts'
      ? (state.cols.title >= 0 ? state.cols.title : state.cols.abstract)
      : header.indexOf('wiki_title');
    var pIdx = header.indexOf('predicted_cosi');
    var html = '<thead><tr><th>#</th><th>' + (m === 'abstracts' ? 'Title' : 'Article') +
      '</th><th>Predicted COSI</th><th class="num">Probability</th></tr></thead><tbody>';
    out.slice(1, 11).forEach(function (r, i) {
      var ok = r[statusIdx] === 'ok';
      var label = r[textIdx] || (m === 'wikipedia' ? r[state.cols.article] : '') || '';
      var prob = ok ? parseFloat(r[header.indexOf('p_' + r[pIdx])]) : null;
      html += '<tr><td class="num">' + (i + 1) + '</td><td class="text">' + escapeHTML(label.slice(0, 120)) + '</td>' +
        (ok
          ? '<td><strong>' + escapeHTML(r[pIdx]) + '</strong></td><td class="num">' + (prob * 100).toFixed(prob < 0.1 ? 1 : 0) + '%</td>'
          : '<td colspan="2" class="err">' + escapeHTML(r[statusIdx]) + '</td>') + '</tr>';
    });
    $('preview').innerHTML = html + '</tbody>';
    $('preview-card').hidden = false;
  }

  var SAMPLES = {
    abstracts: [
      ['title', 'keywords', 'abstract'],
      ['Deep learning prediction of protein-ligand binding affinity', 'drug discovery; graph neural networks; protein structure',
       'We present a graph neural network that predicts binding affinity from protein-ligand complex structures, outperforming docking scores on standard benchmarks.'],
      ['Strain-level profiling of the human gut microbiome', 'metagenomics; microbiome; strain tracking',
       'We introduce a method for strain-resolved metagenomic profiling and apply it to longitudinal stool samples from 300 individuals.']
    ],
    wikipedia: [
      ['url'],
      ['https://en.wikipedia.org/wiki/BLAST_(biotechnology)'],
      ['https://en.wikipedia.org/wiki/Mass_spectrometry'],
      ['AlphaFold'],
      ['Gene regulatory network']
    ]
  };

  // ---------- Wiring ----------

  document.querySelectorAll('input[name="mode"]').forEach(function (el) {
    el.addEventListener('change', function () {
      $('scope-choice').hidden = mode() !== 'wikipedia';
      describe();
    });
  });
  $('file').addEventListener('change', function () { readFile(this.files[0]); });
  var dz = $('dropzone');
  ['dragenter', 'dragover'].forEach(function (t) {
    dz.addEventListener(t, function (e) { e.preventDefault(); dz.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (t) {
    dz.addEventListener(t, function (e) { e.preventDefault(); dz.classList.remove('over'); });
  });
  dz.addEventListener('drop', function (e) { if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]); });
  $('run').addEventListener('click', run);
  $('cancel').addEventListener('click', function () { state.cancelled = true; });
  $('download').addEventListener('click', function () { if (state.output) download(state.outName, toCSV(state.output)); });
  document.querySelectorAll('.sample').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      download('sample_' + a.dataset.sample + '.csv', toCSV(SAMPLES[a.dataset.sample]));
    });
  });
  $('scope-choice').hidden = mode() !== 'wikipedia';
})();
