<?php
// Site configuration for the COSI predictor.
return [
    'model_path' => __DIR__ . '/data/cosi_model.sqlite',

    // COSIs to show. null = every COSI the model knows. To match a given year,
    // list only the COSIs running (percentages are renormalized over this list), e.g.
    // 'active_cosis' => ['3DSIG', 'BOSC', 'MLCSB', 'NetBio', 'RegSys'],
    'active_cosis' => null,

    // Full names shown next to each COSI abbreviation.
    'names' => [
        '3DSIG'        => 'Structural Bioinformatics and Computational Biophysics',
        'BioInfo-Core' => 'Bioinformatics Core Facilities',
        'BioVis'       => 'Biological Data Visualizations',
        'BOKR'         => 'Bio-Ontologies and Knowledge Representation',
        'BOSC'         => 'Bioinformatics Open Source Conference',
        'CAMDA'        => 'Critical Assessment of Massive Data Analysis',
        'CompMS'       => 'Computational Mass Spectrometry',
        'CSI'          => 'Computational Systems Immunology',
        'Education'    => 'Bioinformatics Education',
        'EvolCompGen'  => 'Evolution and Comparative Genomics',
        'Function'     => 'Gene and Protein Function Annotation',
        'HiTSeq'       => 'High-throughput Sequencing Algorithms and Applications',
        'iRNA'         => 'Integrative RNA Biology',
        'MICROBIOME'   => 'Microbiome',
        'MLCSB'        => 'Machine Learning in Computational and Systems Biology',
        'NetBio'       => 'Network Biology',
        'RegSys'       => 'Regulatory and Systems Genomics',
        'SysMod'       => 'Computational Modeling of Biological Systems',
        'TextMining'   => 'Text Mining',
        'TransMed'     => 'Translational Medicine Informatics and Applications',
        'VarI'         => 'Variant Interpretation',
    ],

    'max_chars' => 20000,  // per field

    // Wikimedia asks API clients to identify themselves with contact info (a URL or email).
    'wikipedia_user_agent' => 'COSIPredictor/1.0 (https://cosipredictor.dandeblasio.com)',

    // Linked from the page footer ("Source code on GitHub", "Methods"). '' hides the links.
    'repo_url' => 'https://github.com/danfdeblasio/cosipredictor',
];
