export const opsetVersions = ['7', '8', '9', '10', '11', '12'];

export interface TestSpec {
  name: string;
  opsets?: string[];
  backends?: string[];
}

export type Test = string | TestSpec;

export const enabledTests: Test[] = [
  'test_abs',
  'test_acos',
  'test_acos_example',
  {
    name: 'test_acosh',
    backends: ['CPU', 'WASM'],
    opsets: ['9', '10', '11', '12'],
  },
  {
    name: 'test_acosh_example',
    backends: ['CPU', 'WASM'],
    opsets: ['9', '10', '11', '12'],
  },
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
  {
    name: 'test_argmax_default_axis_example',
    opsets: ['8', '9', '10', '11', '12'],
  },
  {
    name: 'test_argmax_default_axis_random',
    opsets: ['8', '9', '10', '11', '12'],
  },
  {
    name: 'test_argmax_keepdims_example',
    opsets: ['8', '9', '10', '11', '12'],
  },
  {
    name: 'test_argmax_keepdims_random',
    opsets: ['8', '9', '10', '11', '12'],
  },
  {
    name: 'test_argmax_no_keepdims_example',
    opsets: ['8', '9', '10', '11', '12'],
  },
  {
    name: 'test_argmax_no_keepdims_random',
    opsets: ['8', '9', '10', '11', '12'],
  },
  //  "test_argmin_default_axis_example",
  //  "test_argmin_default_axis_random",
  //  "test_argmin_keepdims_example",
  //  "test_argmin_keepdims_random",
  //  "test_argmin_no_keepdims_example",
  //  "test_argmin_no_keepdims_random",
  'test_asin',
  'test_asin_example',
  {
    name: 'test_asinh',
    backends: ['CPU', 'WASM'],
    opsets: ['9', '10', '11', '12'],
  },
  {
    name: 'test_asinh_example',
    backends: ['CPU', 'WASM'],
    opsets: ['9', '10', '11', '12'],
  },
  'test_atan',
  'test_atan_example',
  {
    name: 'test_atanh',
    backends: ['CPU', 'WASM'],
    opsets: ['9', '10', '11', '12'],
  },
  {
    name: 'test_atanh_example',
    backends: ['CPU', 'WASM'],
    opsets: ['9', '10', '11', '12'],
  },
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
  'test_cos',
  'test_cos_example',
  {name: 'test_cosh', opsets: ['9', '10', '11', '12']},
  {name: 'test_cosh_example', opsets: ['9', '10', '11', '12']},
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
  'test_elu',
  'test_elu_default',
  'test_elu_example',
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
  'test_flatten_axis0',
  'test_flatten_axis1',
  'test_flatten_axis2',
  'test_flatten_axis3',
  'test_flatten_default_axis',
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
  'test_hardsigmoid',
  'test_hardsigmoid_default',
  'test_hardsigmoid_example',
  'test_identity',
  {name: 'test_instancenorm_epsilon', opsets: ['9', '10', '11', '12']},
  {name: 'test_instancenorm_example', opsets: ['9', '10', '11', '12']},
  //  "test_isnan",
  'test_leakyrelu',
  'test_leakyrelu_default',
  'test_leakyrelu_example',
  //  "test_less",
  //  "test_less_bcast",
  'test_log',
  'test_log_example',
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
  'test_mean_example',
  'test_mean_one_input',
  'test_mean_two_inputs',
  //  "test_min_example",
  //  "test_min_one_input",
  //  "test_min_two_inputs",
  'test_mul',
  'test_mul_bcast',
  'test_mul_example',
  //  "test_mvn",
  'test_neg',
  'test_neg_example',
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
  'test_pow',
  'test_pow_bcast_array',
  'test_pow_bcast_scalar',
  'test_pow_example',
  'test_prelu_broadcast',
  'test_prelu_example',
  'test_reciprocal',
  {name: 'test_reciprocal_example', backends: ['CPU', 'WASM']},
  'test_reduce_l1_default_axes_keepdims_example',
  'test_reduce_l1_default_axes_keepdims_random',
  'test_reduce_l1_do_not_keepdims_example',
  'test_reduce_l1_do_not_keepdims_random',
  'test_reduce_l1_keep_dims_example',
  'test_reduce_l1_keep_dims_random',
  'test_reduce_l2_default_axes_keepdims_example',
  'test_reduce_l2_default_axes_keepdims_random',
  'test_reduce_l2_do_not_keepdims_example',
  'test_reduce_l2_do_not_keepdims_random',
  'test_reduce_l2_keep_dims_example',
  'test_reduce_l2_keep_dims_random',
  'test_reduce_log_sum',
  'test_reduce_log_sum_asc_axes',
  'test_reduce_log_sum_default',
  'test_reduce_log_sum_desc_axes',
  'test_reduce_log_sum_exp_default_axes_keepdims_example',
  'test_reduce_log_sum_exp_default_axes_keepdims_random',
  'test_reduce_log_sum_exp_do_not_keepdims_example',
  'test_reduce_log_sum_exp_do_not_keepdims_random',
  'test_reduce_log_sum_exp_keepdims_example',
  'test_reduce_log_sum_exp_keepdims_random',
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
  'test_reduce_min_default_axes_keepdims_example',
  'test_reduce_min_default_axes_keepdims_random',
  'test_reduce_min_do_not_keepdims_example',
  'test_reduce_min_do_not_keepdims_random',
  'test_reduce_min_keepdims_example',
  'test_reduce_min_keepdims_random',
  'test_reduce_prod_default_axes_keepdims_example',
  {
    name: 'test_reduce_prod_default_axes_keepdims_random',
    backends: ['CPU'],
  },
  'test_reduce_prod_do_not_keepdims_example',
  'test_reduce_prod_do_not_keepdims_random',
  'test_reduce_prod_keepdims_example',
  'test_reduce_prod_keepdims_random',
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
  'test_selu',
  'test_selu_default',
  'test_selu_example',
  'test_shape',
  'test_shape_example',
  //  "test_shrink_hard",
  //  "test_shrink_soft",
  'test_sigmoid',
  'test_sigmoid_example',
  {
    name: 'test_sign',
    opsets: ['9', '10', '11', '12'],
  },
  //  "test_simple_rnn_defaults",
  //  "test_simple_rnn_with_initial_bias",
  'test_sin',
  'test_sin_example',
  {name: 'test_sinh', opsets: ['9', '10', '11', '12']},
  {name: 'test_sinh_example', opsets: ['9', '10', '11', '12']},
  'test_size',
  'test_size_example',
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
  'test_softplus',
  'test_softplus_example',
  'test_softsign',
  'test_softsign_example',
  //  "test_split_equal_parts_1d",
  //  "test_split_equal_parts_2d",
  //  "test_split_equal_parts_default_axis",
  //  "test_split_variable_parts_1d",
  //  "test_split_variable_parts_2d",
  //  "test_split_variable_parts_default_axis",
  'test_sqrt',
  'test_sqrt_example',
  'test_squeeze',
  'test_sub',
  'test_sub_bcast',
  'test_sub_example',
  'test_sum_example',
  'test_sum_one_input',
  'test_sum_two_inputs',
  'test_tan',
  'test_tan_example',
  {name: 'test_tanh', opsets: ['9', '10', '11', '12']},
  {name: 'test_tanh_example', opsets: ['9', '10', '11', '12']},
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

  // Opset version 10-12
  //'test_adagrad',
  //'test_adagrad_multiple',
  //'test_adam',
  //'test_adam_multiple',
  {
    name: 'test_argmax_default_axis_example_select_last_index',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_default_axis_random_select_last_index',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_keepdims_example_select_last_index',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_keepdims_random_select_last_index',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_negative_axis_keepdims_example',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_negative_axis_keepdims_example_select_last_index',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_negative_axis_keepdims_random',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_negative_axis_keepdims_random_select_last_index',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_no_keepdims_example_select_last_index',
    opsets: ['12'],
  },
  {
    name: 'test_argmax_no_keepdims_random_select_last_index',
    opsets: ['12'],
  },
  //'test_argmin_default_axis_example_select_last_index',
  //'test_argmin_default_axis_random_select_last_index',
  //'test_argmin_keepdims_example_select_last_index',
  //'test_argmin_keepdims_random_select_last_index',
  //'test_argmin_negative_axis_keepdims_example',
  //'test_argmin_negative_axis_keepdims_example_select_last_index',
  //'test_argmin_negative_axis_keepdims_random',
  //'test_argmin_negative_axis_keepdims_random_select_last_index',
  //'test_argmin_no_keepdims_example_select_last_index',
  //'test_argmin_no_keepdims_random_select_last_index',
  //'test_averagepool_2d_ceil',
  //'test_basic_convinteger',
  //'test_bitshift_left_uint16',
  //'test_bitshift_left_uint32',
  //'test_bitshift_left_uint64',
  //'test_bitshift_left_uint8',
  //'test_bitshift_right_uint16',
  //'test_bitshift_right_uint32',
  //'test_bitshift_right_uint64',
  //'test_bitshift_right_uint8',
  {name: 'test_celu', opsets: ['12']},
  {name: 'test_celu_expanded', opsets: ['12']},
  //'test_clip_default_int8_inbounds',
  //'test_clip_default_int8_max',
  //'test_clip_default_int8_min',
  //'test_compress_negative_axis',
  {name: 'test_concat_1d_axis_negative_1', opsets: ['11', '12']},
  {name: 'test_concat_2d_axis_negative_1', opsets: ['11', '12']},
  {name: 'test_concat_2d_axis_negative_2', opsets: ['11', '12']},
  {name: 'test_concat_3d_axis_negative_1', opsets: ['11', '12']},
  {name: 'test_concat_3d_axis_negative_2', opsets: ['11', '12']},
  {name: 'test_concat_3d_axis_negative_3', opsets: ['11', '12']},
  //'test_constantofshape_int_shape_zero',
  //'test_convinteger_with_padding',
  //'test_convtranspose_dilations',
  //'test_cumsum_1d',
  //'test_cumsum_1d_exclusive',
  //'test_cumsum_1d_reverse',
  //'test_cumsum_1d_reverse_exclusive',
  //'test_cumsum_2d_axis_0',
  //'test_cumsum_2d_axis_1',
  //'test_cumsum_2d_negative_axis',
  //'test_depthtospace_crd_mode',
  //'test_depthtospace_crd_mode_example',
  //'test_depthtospace_dcr_mode',
  //'test_dequantizelinear',
  //'test_det_2d',
  //'test_det_nd',
  //'test_dropout_default_mask',
  //'test_dropout_default_mask_ratio',
  //'test_dropout_default_old',
  //'test_dropout_default_ratio',
  //'test_dropout_random_old',
  //'test_dynamicquantizelinear',
  //'test_dynamicquantizelinear_expanded',
  //'test_dynamicquantizelinear_max_adjusted',
  //'test_dynamicquantizelinear_max_adjusted_expanded',
  //'test_dynamicquantizelinear_min_adjusted',
  //'test_dynamicquantizelinear_min_adjusted_expanded',
  //'test_einsum_batch_diagonal',
  //'test_einsum_batch_matmul',
  //'test_einsum_inner_prod',
  //'test_einsum_sum',
  //'test_einsum_transpose',
  {name: 'test_flatten_negative_axis1', opsets: ['11', '12']},
  {name: 'test_flatten_negative_axis2', opsets: ['11', '12']},
  {name: 'test_flatten_negative_axis3', opsets: ['11', '12']},
  {name: 'test_flatten_negative_axis4', opsets: ['11', '12']},
  //'test_gather_elements_0',
  //'test_gather_elements_1',
  //'test_gather_elements_negative_indices',
  //'test_gathernd_example_float32',
  //'test_gathernd_example_int32',
  //'test_gathernd_example_int32_batch_dim1',
  //'test_gather_negative_indices',
  {name: 'test_gemm_all_attributes', opsets: ['11', '12']},
  {name: 'test_gemm_alpha', opsets: ['11', '12']},
  {name: 'test_gemm_beta', opsets: ['11', '12']},
  {name: 'test_gemm_default_matrix_bias', opsets: ['11', '12']},
  {name: 'test_gemm_default_no_bias', opsets: ['11', '12']},
  {name: 'test_gemm_default_scalar_bias', opsets: ['11', '12']},
  {name: 'test_gemm_default_single_elem_vector_bias', opsets: ['11', '12']},
  {name: 'test_gemm_default_vector_bias', opsets: ['11', '12']},
  {name: 'test_gemm_default_zero_bias', opsets: ['11', '12']},
  {name: 'test_gemm_transposeA', opsets: ['11', '12']},
  {name: 'test_gemm_transposeB', opsets: ['11', '12']},
  //'test_greater_equal',
  //'test_greater_equal_bcast',
  //'test_greater_equal_bcast_expanded',
  //'test_greater_equal_expanded',
  //'test_hardmax_negative_axis',
  //'test_isinf',
  //'test_isinf_negative',
  //'test_isinf_positive',
  //'test_less_equal',
  //'test_less_equal_bcast',
  //'test_less_equal_bcast_expanded',
  //'test_less_equal_expanded',
  //'test_logsoftmax_negative_axis',
  //'test_matmulinteger',
  //'test_max_float16',
  //'test_max_float32',
  //'test_max_float64',
  //'test_max_int16',
  //'test_max_int32',
  //'test_max_int64',
  //'test_max_int8',
  //'test_maxpool_2d_ceil',
  //'test_maxpool_2d_dilations',
  //'test_maxpool_2d_uint8',
  //'test_max_uint16',
  //'test_max_uint32',
  //'test_max_uint64',
  //'test_max_uint8',
  //'test_min_float16',
  //'test_min_float32',
  //'test_min_float64',
  //'test_min_int16',
  //'test_min_int32',
  //'test_min_int64',
  //'test_min_int8',
  //'test_min_uint16',
  //'test_min_uint32',
  //'test_min_uint64',
  //'test_min_uint8',
  //'test_mod_broadcast',
  //'test_mod_int64_fmod',
  //'test_mod_mixed_sign_float16',
  //'test_mod_mixed_sign_float32',
  //'test_mod_mixed_sign_float64',
  //'test_mod_mixed_sign_int16',
  //'test_mod_mixed_sign_int32',
  //'test_mod_mixed_sign_int64',
  //'test_mod_mixed_sign_int8',
  //'test_mod_uint16',
  //'test_mod_uint32',
  //'test_mod_uint64',
  //'test_mod_uint8',
  //'test_momentum',
  //'test_momentum_multiple',
  //'test_mvn_expanded',
  //'test_negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index',
  //'test_negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NC',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1_weight',
  //'test_negative_log_likelihood_loss_input_shape_is_NCd1_weight_expanded',
  //'test_negative_log_likelihood_loss_input_shape_is_NC_expanded',
  //'test_nesterov_momentum',
  //'test_nonmaxsuppression_center_point_box_format',
  //'test_nonmaxsuppression_flipped_coordinates',
  //'test_nonmaxsuppression_identical_boxes',
  //'test_nonmaxsuppression_limit_output_size',
  //'test_nonmaxsuppression_single_box',
  //'test_nonmaxsuppression_suppress_by_IOU',
  //'test_nonmaxsuppression_suppress_by_IOU_and_scores',
  //'test_nonmaxsuppression_two_batches',
  //'test_nonmaxsuppression_two_classes',
  //'test_onehot_negative_indices',
  //'test_onehot_with_negative_axis',
  //'test_pow_types_float',
  //'test_pow_types_float32_int32',
  //'test_pow_types_float32_int64',
  //'test_pow_types_float32_uint32',
  //'test_pow_types_float32_uint64',
  //'test_pow_types_int',
  //'test_pow_types_int32_float32',
  //'test_pow_types_int32_int32',
  //'test_pow_types_int64_float32',
  //'test_pow_types_int64_int64',
  //'test_qlinearconv',
  //'test_qlinearmatmul_2D',
  //'test_qlinearmatmul_3D',
  //'test_quantizelinear',
  //'test_range_float_type_positive_delta',
  //'test_range_float_type_positive_delta_expanded',
  //'test_range_int32_type_negative_delta',
  //'test_range_int32_type_negative_delta_expanded',
  {
    name: 'test_reduce_l1_negative_axes_keep_dims_example',
    opsets: ['11', '12'],
  },
  {name: 'test_reduce_l1_negative_axes_keep_dims_random', opsets: ['11', '12']},
  {
    name: 'test_reduce_l2_negative_axes_keep_dims_example',
    opsets: ['11', '12'],
  },
  {name: 'test_reduce_l2_negative_axes_keep_dims_random', opsets: ['11', '12']},
  {
    name: 'test_reduce_log_sum_exp_negative_axes_keepdims_example',
    opsets: ['11', '12'],
  },
  {
    name: 'test_reduce_log_sum_exp_negative_axes_keepdims_random',
    opsets: ['11', '12'],
  },
  {name: 'test_reduce_log_sum_negative_axes', opsets: ['11', '12']},
  {
    name: 'test_reduce_max_negative_axes_keepdims_example',
    opsets: ['11', '12'],
  },
  {name: 'test_reduce_max_negative_axes_keepdims_random', opsets: ['11', '12']},
  {
    name: 'test_reduce_mean_negative_axes_keepdims_example',
    opsets: ['11', '12'],
  },
  {
    name: 'test_reduce_mean_negative_axes_keepdims_random',
    opsets: ['11', '12'],
  },
  {
    name: 'test_reduce_min_negative_axes_keepdims_example',
    opsets: ['11', '12'],
  },
  {name: 'test_reduce_min_negative_axes_keepdims_random', opsets: ['11', '12']},
  {
    name: 'test_reduce_prod_negative_axes_keepdims_example',
    opsets: ['11', '12'],
  },
  {
    name: 'test_reduce_prod_negative_axes_keepdims_random',
    opsets: ['11', '12'],
  },
  {
    name: 'test_reduce_sum_negative_axes_keepdims_example',
    opsets: ['11', '12'],
  },
  {name: 'test_reduce_sum_negative_axes_keepdims_random', opsets: ['11', '12']},
  {
    name: 'test_reduce_sum_square_negative_axes_keepdims_example',
    opsets: ['11', '12'],
  },
  {
    name: 'test_reduce_sum_square_negative_axes_keepdims_random',
    opsets: ['11', '12'],
  },
  //'test_reshape_negative_extended_dims',
  //'test_reshape_reordered_all_dims',
  //'test_reshape_reordered_last_dims',
  //'test_reshape_zero_and_negative_dim',
  //'test_reshape_zero_dim',
  //'test_resize_downsample_scales_cubic',
  //'test_resize_downsample_scales_cubic_align_corners',
  //'test_resize_downsample_scales_cubic_A_n0p5_exclude_outside',
  //'test_resize_downsample_scales_linear',
  //'test_resize_downsample_scales_linear_align_corners',
  //'test_resize_downsample_scales_nearest',
  //'test_resize_downsample_sizes_cubic',
  //'test_resize_downsample_sizes_linear_pytorch_half_pixel',
  //'test_resize_downsample_sizes_nearest',
  //'test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn',
  //'test_resize_tf_crop_and_resize',
  //'test_resize_upsample_scales_cubic',
  //'test_resize_upsample_scales_cubic_align_corners',
  //'test_resize_upsample_scales_cubic_A_n0p5_exclude_outside',
  //'test_resize_upsample_scales_cubic_asymmetric',
  //'test_resize_upsample_scales_linear',
  //'test_resize_upsample_scales_linear_align_corners',
  //'test_resize_upsample_scales_nearest',
  //'test_resize_upsample_sizes_cubic',
  //'test_resize_upsample_sizes_nearest',
  //'test_resize_upsample_sizes_nearest_ceil_half_pixel',
  //'test_resize_upsample_sizes_nearest_floor_align_corners',
  //'test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric',
  //'test_reversesequence_batch',
  //'test_reversesequence_time',
  //'test_roialign',
  //'test_round',
  //'test_scatter_elements_with_axis',
  //'test_scatter_elements_with_negative_indices',
  //'test_scatter_elements_without_axis',
  //'test_scatternd',
  {name: 'test_slice_default_steps', opsets: ['10', '11', '12']},
  {name: 'test_slice_negative_axes', opsets: ['11', '12']},
  {name: 'test_slice_neg_steps', opsets: ['10', '11', '12']},
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob',
  //'test_softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index',
  //'test_softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded',
  //'test_softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob',
  //'test_softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob_expanded',
  //'test_softmax_cross_entropy_mean',
  //'test_softmax_cross_entropy_mean_3d',
  //'test_softmax_cross_entropy_mean_3d_expanded',
  //'test_softmax_cross_entropy_mean_3d_log_prob',
  //'test_softmax_cross_entropy_mean_3d_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_expanded',
  //'test_softmax_cross_entropy_mean_log_prob',
  //'test_softmax_cross_entropy_mean_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_3d',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_3d_expanded',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_4d',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_4d_expanded',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_expanded',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_log_prob',
  //'test_softmax_cross_entropy_mean_no_weight_ignore_index_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_weight',
  //'test_softmax_cross_entropy_mean_weight_expanded',
  //'test_softmax_cross_entropy_mean_weight_ignore_index',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_3d',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_3d_expanded',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_4d',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_4d_expanded',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_expanded',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_log_prob',
  //'test_softmax_cross_entropy_mean_weight_ignore_index_log_prob_expanded',
  //'test_softmax_cross_entropy_mean_weight_log_prob',
  //'test_softmax_cross_entropy_mean_weight_log_prob_expanded',
  //'test_softmax_cross_entropy_none',
  //'test_softmax_cross_entropy_none_expanded',
  //'test_softmax_cross_entropy_none_log_prob',
  //'test_softmax_cross_entropy_none_log_prob_expanded',
  //'test_softmax_cross_entropy_none_weights',
  //'test_softmax_cross_entropy_none_weights_expanded',
  //'test_softmax_cross_entropy_none_weights_log_prob',
  //'test_softmax_cross_entropy_none_weights_log_prob_expanded',
  //'test_softmax_cross_entropy_sum',
  //'test_softmax_cross_entropy_sum_expanded',
  //'test_softmax_cross_entropy_sum_log_prob',
  //'test_softmax_cross_entropy_sum_log_prob_expanded',
  //'test_softmax_negative_axis',
  //'test_split_zero_size_splits',
  //'test_squeeze_negative_axes',
  //'test_strnormalizer_export_monday_casesensintive_lower',
  //'test_strnormalizer_export_monday_casesensintive_nochangecase',
  //'test_strnormalizer_export_monday_casesensintive_upper',
  //'test_strnormalizer_export_monday_empty_output',
  //'test_strnormalizer_export_monday_insensintive_upper_twodim',
  //'test_strnormalizer_nostopwords_nochangecase',
  //'test_top_k_negative_axis',
  //'test_top_k_smallest',
  //'test_training_dropout',
  //'test_training_dropout_default',
  //'test_training_dropout_default_mask',
  //'test_training_dropout_mask',
  //'test_training_dropout_zero_ratio',
  //'test_training_dropout_zero_ratio_mask',
  //'test_unique_not_sorted_without_axis',
  //'test_unique_sorted_with_axis',
  //'test_unique_sorted_with_axis_3d',
  //'test_unique_sorted_with_negative_axis',
  //'test_unique_sorted_without_axis',
  //'test_unsqueeze_axis_0',
  //'test_unsqueeze_axis_1',
  //'test_unsqueeze_axis_2',
  //'test_unsqueeze_axis_3',
  //'test_unsqueeze_negative_axes',
  //'test_unsqueeze_three_axes',
  //'test_unsqueeze_two_axes',
  //'test_unsqueeze_unsorted_axes',
  //'test_where_long_example';
];
