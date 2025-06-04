<table border="1" style="text-align:center; border-collapse:collapse; width: 100%;">
  <thead>
    <tr>
      <th rowspan="2">Task Type</th>
      <th rowspan="2">Task Name</th>
      <th rowspan="2">Testing File</th>
      <th colspan="3">F1 Score</th>
      <th colspan="3">ROUGE-L</th>
    </tr>
    <tr>
      <th>Baseline</th>
      <th>Stage 1</th>
      <th>Stage 2</th>
      <th>Baseline</th>
      <th>Stage 1</th>
      <th>Stage 2</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">Sequence Understanding</td><td rowspan="2">EC Number Prediction (split100)</td>
      <td>CLEAN_EC_number_new_test</td><td>0.3468</td><td>0.2278</td><td>0.0000</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td><td>0.0738</td><td>0.3320</td><td>0.0000</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Fold Type Prediction</td>
      <td>fold_type_test_Fold_Holdout</td><td>0.1045</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Superfamily_Holdout</td><td>0.1507</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Family_Holdout</td><td>0.6145</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>Subcellular Localization</td><td>subcell_loc_test</td><td>0.4214</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>

    <tr><td rowspan="9">Annotation Prediction</td><td rowspan="3">Function Keywords Prediction</td>
      <td>CASPSimilarSeq_keywords_test</td><td>0.4385</td><td>0.1335</td><td>0.3225</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_keywords_test</td><td>0.6650</td><td>0.0981</td><td>0.0842</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_keywords_test</td><td>0.7374</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>

    <tr><td rowspan="3">GO Term Prediction</td>
      <td>CASPSimilarSeq_go_terms_test</td><td>0.0990</td><td>0.7608</td><td>0.6894</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_go_terms_test</td><td>0.6304</td><td>0.6748</td><td>0.6926</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_go_terms_test</td><td>0.6849</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>

    <tr><td rowspan="3">Function Description</td>
      <td>CASPSimilarSeq_function_test</td><td>-</td><td>-</td><td>-</td><td>0.7524</td><td>0.0191</td><td>0.0304</td>
    </tr>
    <tr>
      <td>IDFilterSeq_function_test</td><td>-</td><td>-</td><td>-</td><td>0.4786</td><td>0.0200</td><td>0.0237</td>
    </tr>
    <tr>
      <td>UniProtSeq_function_test</td><td>-</td><td>-</td><td>-</td><td>0.5144</td><td>-</td><td>-</td>
    </tr>

    <tr><td rowspan="3">Knowledge Mining</td><td>Tissue Location from Gene Symbol</td>
      <td>gene_symbol_to_tissue_test</td><td>0.5466</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Symbol</td><td>gene_symbol_to_cancer_test</td>
      <td>0.2664</td><td>0.0000</td><td>0.0000</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Name</td><td>gene_name_to_cancer_test</td>
      <td>0.2659</td><td>0.0000</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
  </tbody>
</table>
