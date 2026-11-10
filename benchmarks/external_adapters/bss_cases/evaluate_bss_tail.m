function evaluate_bss_tail(bss_root, result_file, input_file, output_file)
addpath(fullfile(bss_root, 'stk'));
stk_init;
saved = load(result_file, 'stage_data');
inputs = load(input_file, 'x');
final = saved.stage_data(end);
prediction = stk_predict(final.model, final.observation_x, final.observation_y, inputs.x);
response = -double(prediction.mean);
assert(all(isfinite(response)), 'Nonfinite tail predictions.');
handle = fopen(output_file, 'w');
assert(handle >= 0, 'Cannot open prediction output.');
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '%s', jsonencode(response));
end
