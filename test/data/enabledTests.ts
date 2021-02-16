export const opsetVersions = ['7', '8', '9', '10', '11', '12'];

export interface TestSpec {
  name: string;
  opsets?: string[];
}

export type Test = string | TestSpec;

export const enabledTests: Test[] = [
  //  "test_abs",
  //  "test_acos",
  //  "test_acos_example",
  //  "test_acosh",
  //  "test_acosh_example",
  'test_add',
  'test_add_bcast',
  //  "test_and2d",
  //  "test_and3d",
  //  "test_and4d",
  //  "test_and_bcast3v1d",
  //  "test_and_bcast3v2d",
  //  "test_and_bcast4v2d",
  //  "test_and_bcast4v3d",
  // "test_and_bcast4v4d",
  //  "test_argmax_default_axis_example",
  //  "test_argmax_default_axis_random",
  //  "test_argmax_keepdims_example",
  //  "test_argmax_keepdims_random",
  //  "test_argmax_no_keepdims_example",
  //  "test_argmax_no_keepdims_random",
  //  "test_argmin_default_axis_example",
  //  "test_argmin_default_axis_random",
  //  "test_argmin_keepdims_example",
  //  "test_argmin_keepdims_random",
  //  "test_argmin_no_keepdims_example",
  //  "test_argmin_no_keepdims_random",
  //  "test_asin",
  //  "test_asin_example",
  //  "test_asinh",
  //  "test_asinh_example",
  //  "test_atan",
  //  "test_atan_example",
  //  "test_atanh",
  //  "test_atanh_example",
  //  "test_averagepool_1d_default",
  //  "test_averagepool_2d_default",
  //  "test_averagepool_2d_pads",
  //  "test_averagepool_2d_pads_count_include_pad",
  //  "test_averagepool_2d_precomputed_pads",
  //  "test_averagepool_2d_precomputed_pads_count_include_pad",
  //  "test_averagepool_2d_precomputed_same_upper",
  //  "test_averagepool_2d_precomputed_strides",
  //  "test_averagepool_2d_same_lower",
  //  "test_averagepool_2d_same_upper",
  //  "test_averagepool_2d_strides",
  //  "test_averagepool_3d_default",
  'test_basic_conv_without_padding',
  'test_basic_conv_with_padding',
  {name: 'test_batchnorm_epsilon', opsets: ['9', '10']},
  {name: 'test_batchnorm_example', opsets: ['9', '10']},
  //  "test_cast_DOUBLE_to_FLOAT",
  //  "test_cast_DOUBLE_to_FLOAT16",
  //  "test_cast_FLOAT16_to_DOUBLE",
  //  "test_cast_FLOAT16_to_FLOAT",
  //  "test_cast_FLOAT_to_DOUBLE",
  //  "test_cast_FLOAT_to_FLOAT16",
  //  "test_cast_FLOAT_to_STRING",
  //  "test_cast_STRING_to_FLOAT",
  'test_ceil',
  'test_ceil_example',
  {name: 'test_clip', opsets: ['7', '8', '9', '10']},
  {name: 'test_clip_default_inbounds', opsets: ['9', '10']},
  {name: 'test_clip_default_max', opsets: ['7', '8', '9', '10']},
  {name: 'test_clip_default_min', opsets: ['7', '8', '9', '10']},
  {name: 'test_clip_example', opsets: ['7', '8', '9', '10']},
  {name: 'test_clip_inbounds', opsets: ['9', '10']},
  {name: 'test_clip_outbounds', opsets: ['9', '10']},
  {name: 'test_clip_splitbounds', opsets: ['9', '10']},
  //  "test_compress_0",
  //  "test_compress_1",
  //  "test_compress_default_axis",
  'test_concat_1d_axis_0',
  'test_concat_2d_axis_0',
  'test_concat_2d_axis_1',
  'test_concat_3d_axis_0',
  'test_concat_3d_axis_1',
  'test_concat_3d_axis_2',
  'test_constant',
  {name: 'test_constantofshape_float_ones', opsets: ['9', '10', '11', '12']},
  //  "test_constantofshape_int_zeros",
  {name: 'test_constant_pad', opsets: ['7', '8', '9', '10']},
  //  "test_convtranspose",
  //  "test_convtranspose_1d",
  //  "test_convtranspose_3d",
  //  "test_convtranspose_kernel_shape",
  //  "test_convtranspose_output_shape",
  //  "test_convtranspose_pad",
  //  "test_convtranspose_pads",
  //  "test_convtranspose_with_kernel",
  'test_conv_with_strides_and_asymmetric_padding',
  'test_conv_with_strides_no_padding',
  'test_conv_with_strides_padding',
  //  "test_cos",
  //  "test_cos_example",
  //  "test_cosh",
  //  "test_cosh_example",
  //  "test_depthtospace",
  //  "test_depthtospace_example",
  'test_div',
  'test_div_bcast',
  'test_div_example',
  //  "test_dropout_default",
  //  "test_dropout_random",
  //  "test_dynamic_slice",
  //  "test_dynamic_slice_default_axes",
  //  "test_dynamic_slice_end_out_of_bounds",
  //  "test_dynamic_slice_neg",
  //  "test_dynamic_slice_start_out_of_bounds",
  {name: 'test_edge_pad', opsets: ['7', '8', '9', '10']},
  //  "test_elu",
  //  "test_elu_default",
  //  "test_elu_example",
  //  "test_equal",
  //  "test_equal_bcast",
  // "test_erf",
  'test_exp',
  {name: 'test_expand_dim_changed', opsets: ['9', '10', '11', '12']},
  {name: 'test_expand_dim_unchanged', opsets: ['9', '10', '11', '12']},
  'test_exp_example',
  // "test_eyelike_populate_off_main_diagonal",
  // "test_eyelike_with_dtype",
  // "test_eyelike_without_dtype",
  // "test_flatten_axis0",
  // "test_flatten_axis1",
  //  "test_flatten_axis2",
  //  "test_flatten_axis3",
  //  "test_flatten_default_axis",
  'test_floor',
  'test_floor_example',
  'test_gather_0',
  'test_gather_1',
  {name: 'test_gemm_broadcast', opsets: ['7', '8', '9', '10']},
  {name: 'test_gemm_nobroadcast', opsets: ['7', '8', '9', '10']},
  'test_globalaveragepool',
  'test_globalaveragepool_precomputed',
  //  "test_globalmaxpool",
  // "test_globalmaxpool_precomputed",
  //  "test_greater",
  //  "test_greater_bcast",
  //  "test_gru_defaults",
  //  "test_gru_seq_length",
  //  "test_gru_with_initial_bias",
  //  "test_hardmax_axis_0",
  //  "test_hardmax_axis_1",
  //  "test_hardmax_axis_2",
  //  "test_hardmax_default_axis",
  //  "test_hardmax_example",
  //  "test_hardmax_one_hot",
  //  "test_hardsigmoid",
  //  "test_hardsigmoid_default",
  //  "test_hardsigmoid_example",
  //  "test_identity",
  {name: 'test_instancenorm_epsilon', opsets: ['9', '10', '11', '12']},
  {name: 'test_instancenorm_example', opsets: ['9', '10', '11', '12']},
  //  "test_isnan",
  //  "test_leakyrelu",
  //  "test_leakyrelu_default",
  //  "test_leakyrelu_example",
  //  "test_less",
  //  "test_less_bcast",
  //  "test_log",
  //  "test_log_example",
  //  "test_logsoftmax_axis_0",
  //  "test_logsoftmax_axis_1",
  //  "test_logsoftmax_axis_2",
  //  "test_logsoftmax_default_axis",
  //  "test_logsoftmax_example_1",
  //  "test_logsoftmax_large_number",
  //  "test_lrn",
  //  "test_lrn_default",
  //  "test_lstm_defaults",
  //  "test_lstm_with_initial_bias",
  //  "test_lstm_with_peepholes",
  'test_matmul_2d',
  'test_matmul_3d',
  'test_matmul_4d',
  //  "test_max_example",
  //  "test_max_one_input",
  //  "test_maxpool_1d_default",
  //  "test_maxpool_2d_default",
  //  "test_maxpool_2d_pads",
  //  "test_maxpool_2d_precomputed_pads",
  //  "test_maxpool_2d_precomputed_same_upper",
  //  "test_maxpool_2d_precomputed_strides",
  //  "test_maxpool_2d_same_lower",
  //  "test_maxpool_2d_same_upper",
  //  "test_maxpool_2d_strides",
  //  "test_maxpool_3d_default",
  //  "test_maxpool_with_argmax_2d_precomputed_pads",
  // "test_maxpool_with_argmax_2d_precomputed_strides",
  //  "test_max_two_inputs",
  //  "test_maxunpool_export_without_output_shape",
  //  "test_maxunpool_export_with_output_shape",
  //  "test_mean_example",
  //  "test_mean_one_input",
  //  "test_mean_two_inputs",
  //  "test_min_example",
  //  "test_min_one_input",
  //  "test_min_two_inputs",
  'test_mul',
  'test_mul_bcast',
  'test_mul_example',
  //  "test_mvn",
  //  "test_neg",
  //  "test_neg_example",
  //  "test_nonzero_example",
  //  "test_not_2d",
  //  "test_not_3d",
  //  "test_not_4d",
  //  "test_onehot_with_axis",
  //  "test_onehot_without_axis",
  //  "test_or2d",
  //  "test_or3d",
  //  "test_or4d",
  //  "test_or_bcast3v1d",
  //  "test_or_bcast3v2d",
  //  "test_or_bcast4v2d",
  //  "test_or_bcast4v3d",
  //  "test_or_bcast4v4d",
  //  "test_pow",
  //  "test_pow_bcast_array",
  //  "test_pow_bcast_scalar",
  //  "test_pow_example",
  //  "test_prelu_broadcast",
  //  "test_prelu_example",
  //  "test_reciprocal",
  //  "test_reciprocal_example",
  //  "test_reduce_l1_default_axes_keepdims_example",
  //  "test_reduce_l1_default_axes_keepdims_random",
  //  "test_reduce_l1_do_not_keepdims_example",
  //  "test_reduce_l1_do_not_keepdims_random",
  //  "test_reduce_l1_keep_dims_example",
  //  "test_reduce_l1_keep_dims_random",
  //  "test_reduce_l2_default_axes_keepdims_example",
  //  "test_reduce_l2_default_axes_keepdims_random",
  //  "test_reduce_l2_do_not_keepdims_example",
  //  "test_reduce_l2_do_not_keepdims_random",
  //  "test_reduce_l2_keep_dims_example",
  //  "test_reduce_l2_keep_dims_random",
  //  "test_reduce_log_sum",
  //  "test_reduce_log_sum_asc_axes",
  //  "test_reduce_log_sum_default",
  //  "test_reduce_log_sum_desc_axes",
  //  "test_reduce_log_sum_exp_default_axes_keepdims_example",
  //  "test_reduce_log_sum_exp_default_axes_keepdims_random",
  //  "test_reduce_log_sum_exp_do_not_keepdims_example",
  //  "test_reduce_log_sum_exp_do_not_keepdims_random",
  //  "test_reduce_log_sum_exp_keepdims_example",
  //  "test_reduce_log_sum_exp_keepdims_random",
  'test_reduce_max_default_axes_keepdim_example',
  'test_reduce_max_default_axes_keepdims_random',
  'test_reduce_max_do_not_keepdims_example',
  'test_reduce_max_do_not_keepdims_random',
  'test_reduce_max_keepdims_example',
  'test_reduce_max_keepdims_random',
  'test_reduce_mean_default_axes_keepdims_example',
  'test_reduce_mean_default_axes_keepdims_random',
  'test_reduce_mean_do_not_keepdims_example',
  'test_reduce_mean_do_not_keepdims_random',
  'test_reduce_mean_keepdims_example',
  'test_reduce_mean_keepdims_random',
  //  "test_reduce_min_default_axes_keepdims_example",
  //  "test_reduce_min_default_axes_keepdims_random",
  //  "test_reduce_min_do_not_keepdims_example",
  //  "test_reduce_min_do_not_keepdims_random",
  //  "test_reduce_min_keepdims_example",
  //  "test_reduce_min_keepdims_random",
  //  "test_reduce_prod_default_axes_keepdims_example",
  //  "test_reduce_prod_default_axes_keepdims_random",
  //  "test_reduce_prod_do_not_keepdims_example",
  // "test_reduce_prod_do_not_keepdims_random",
  // "test_reduce_prod_keepdims_example",
  //  "test_reduce_prod_keepdims_random",
  'test_reduce_sum_default_axes_keepdims_example',
  'test_reduce_sum_default_axes_keepdims_random',
  'test_reduce_sum_do_not_keepdims_example',
  'test_reduce_sum_do_not_keepdims_random',
  'test_reduce_sum_keepdims_example',
  'test_reduce_sum_keepdims_random',
  'test_reduce_sum_square_default_axes_keepdims_example',
  'test_reduce_sum_square_default_axes_keepdims_random',
  'test_reduce_sum_square_do_not_keepdims_example',
  'test_reduce_sum_square_do_not_keepdims_random',
  'test_reduce_sum_square_keepdims_example',
  'test_reduce_sum_square_keepdims_random',
  {name: 'test_reflect_pad', opsets: ['7', '8', '9', '10']},
  'test_relu',
  'test_reshape_extended_dims',
  'test_reshape_negative_dim',
  'test_reshape_one_dim',
  'test_reshape_reduced_dims',
  {name: 'test_reshape_reordered_dims', opsets: ['7', '8', '9', '10']},
  // "test_rnn_seq_length",
  //  "test_scan9_sum",
  //  "test_scan_sum",
  //  "test_scatter_with_axis",
  //  "test_scatter_without_axis",
  //  "test_selu",
  //  "test_selu_default",
  //  "test_selu_example",
  'test_shape',
  'test_shape_example',
  //  "test_shrink_hard",
  //  "test_shrink_soft",
  //  "test_sigmoid",
  //  "test_sigmoid_example",
  //  "test_sign",
  //  "test_simple_rnn_defaults",
  //  "test_simple_rnn_with_initial_bias",
  //  "test_sin",
  //  "test_sin_example",
  //  "test_sinh",
  //  "test_sinh_example",
  //  "test_size",
  //  "test_size_example",
  {name: 'test_slice', opsets: ['7', '8', '9']},
  {name: 'test_slice_default_axes', opsets: ['7', '8', '9']},
  {name: 'test_slice_end_out_of_bounds', opsets: ['7', '8', '9']},
  {name: 'test_slice_neg', opsets: ['7', '8', '9']},
  {name: 'test_slice_start_out_of_bounds', opsets: ['7', '8', '9']},
  'test_softmax_axis_0',
  'test_softmax_axis_1',
  'test_softmax_axis_2',
  'test_softmax_default_axis',
  'test_softmax_example',
  'test_softmax_large_number',
  //  "test_softplus",
  //  "test_softplus_example",
  //  "test_softsign",
  //  "test_softsign_example",
  //  "test_split_equal_parts_1d",
  //  "test_split_equal_parts_2d",
  //  "test_split_equal_parts_default_axis",
  //  "test_split_variable_parts_1d",
  //  "test_split_variable_parts_2d",
  //  "test_split_variable_parts_default_axis",
  //  "test_sqrt",
  //  "test_sqrt_example",
  //  "test_squeeze",
  'test_sub',
  'test_sub_bcast',
  'test_sub_example',
  //  "test_sum_example",
  //  "test_sum_one_input",
  //  "test_sum_two_inputs",
  //  "test_tan",
  //  "test_tan_example",
  //  "test_tanh",
  //  "test_tanh_example",
  //  "test_tfidfvectorizer_tf_batch_onlybigrams_skip0",
  //  "test_tfidfvectorizer_tf_batch_onlybigrams_skip5",
  //  "test_tfidfvectorizer_tf_batch_uniandbigrams_skip5",
  //  "test_tfidfvectorizer_tf_onlybigrams_levelempty",
  //  "test_tfidfvectorizer_tf_only_bigrams_skip0",
  //  "test_tfidfvectorizer_tf_onlybigrams_skip5",
  //  "test_tfidfvectorizer_tf_uniandbigrams_skip5",
  //  "test_thresholdedrelu",
  //  "test_thresholdedrelu_default",
  //  "test_thresholdedrelu_example",
  'test_tile',
  'test_tile_precomputed',
  //  "test_top_k",
  'test_transpose_all_permutations_0',
  'test_transpose_all_permutations_1',
  'test_transpose_all_permutations_2',
  'test_transpose_all_permutations_3',
  'test_transpose_all_permutations_4',
  'test_transpose_all_permutations_5',
  'test_transpose_default',
  {name: 'test_unsqueeze', opsets: ['7', '8', '9', '10']},
  'test_upsample_nearest',
  //  "test_where_example",
  //  "test_xor2d",
  //  "test_xor3d",
  //  "test_xor4d",
  //  "test_xor_bcast3v1d",
  //  "test_xor_bcast3v2d",
  //  "test_xor_bcast4v2d",
  //  "test_xor_bcast4v3d",
  //  "test_xor_bcast4v4d",
];