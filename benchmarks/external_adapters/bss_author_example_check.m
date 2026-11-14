function report = bss_author_example_check(output_json)
%BSS_AUTHOR_EXAMPLE_CHECK Run one pinned authors' BSS numerical example.
%
% This is an acceptance check, not a project-case wrapper. It executes the
% official case #006 and official bss() implementation with default BSS
% options, figures disabled, and the reproducible MATLAB seed suggested in
% the authors' test script. No algorithm code is copied or modified here.

expected_commit = '574e083634b9d640c4e6bdfc6b94f923e074ff82';
adapter_root = fileparts(mfilename('fullpath'));
moretest_root = fileparts(adapter_root);
bss_root = fullfile(moretest_root, 'third_party', 'bss');

assert(isfolder(bss_root), 'Pinned BSS checkout is absent: %s', bss_root);
[git_status, git_output] = system(sprintf('git -C "%s" rev-parse HEAD', bss_root));
assert(git_status == 0, 'Cannot verify BSS git checkout: %s', git_output);
actual_commit = strtrim(git_output);
assert(strcmp(actual_commit, expected_commit), ...
    'BSS commit mismatch: expected %s, found %s', expected_commit, actual_commit);

addpath(fullfile(bss_root, 'scripts'));
addpath(fullfile(bss_root, 'examples'));
addpath(fullfile(bss_root, 'functions'));
addpath(fullfile(bss_root, 'functions', 'distrib'));
addpath(fullfile(bss_root, 'functions', 'bss'));
addpath(fullfile(bss_root, 'stk'));
stk_init;

mex_path = fullfile(bss_root, 'functions', 'bss', ...
    ['replicate_particles.' mexext]);
assert(isfile(mex_path), 'Compiled BSS helper is absent: %s', mex_path);

rng(0, 'twister');
case_number = 6;
problem = select_example(case_number);
author_cost_fun = problem.cost_fun;
raw_cost_fun_call_count = 0;
raw_cost_fun_row_count = 0;
problem.cost_fun = @counted_author_cost_fun;
options = bss_get_default_options([], problem);
options.figs.disable_all = true;

started = tic;
stage_data = bss(problem, options);
elapsed_seconds = toc(started);

pf_estimate = stage_data(end).Pf;
pf_reference = problem.Pf_true;
reference_ratio = pf_estimate / pf_reference;
stage_evaluations = [stage_data.n_evals];
execution_pass = isfinite(pf_estimate) && pf_estimate > 0 && ...
    ~isempty(stage_data) && all(stage_evaluations >= 0);
reference_consistency_pass = reference_ratio >= 0.5 && reference_ratio <= 2.0;

report = struct();
if execution_pass && reference_consistency_pass
    report.status = 'PASS';
else
    report.status = 'FAIL';
end
report.check = 'authors-official-case-006';
report.third_party_repository = ...
    'https://git.code.sf.net/p/kriging/contrib-bss';
report.third_party_commit = actual_commit;
report.third_party_api = 'functions/bss/bss.m';
report.license = 'LGPL-2.1-or-later';
report.matlab_version = version;
report.matlab_release = version('-release');
report.stk_version = stk_version();
report.stk_archive = 'pinned release';
report.random_seed = 0;
report.random_generator = 'twister';
report.case_number = case_number;
report.case_name = problem.name;
report.dimension = problem.dim;
report.pf_estimate = pf_estimate;
report.pf_reference = pf_reference;
report.reference_ratio = reference_ratio;
report.reference_acceptance_interval = [0.5 2.0];
report.execution_pass = execution_pass;
report.reference_consistency_pass = reference_consistency_pass;
report.number_of_stages = numel(stage_data);
report.initial_design_size = options.N0;
report.stage_new_evaluations = stage_evaluations;
report.author_decision_label_count = options.N0 + sum(stage_evaluations);
report.author_raw_cost_fun_call_count = raw_cost_fun_call_count;
report.author_raw_cost_fun_row_count = raw_cost_fun_row_count;
report.cost_accounting_note = [ ...
    'author_decision_label_count is N0 plus SUR-selected labels only; ' ...
    'author_raw_cost_fun_row_count also includes full SMC-population truth ' ...
    'calls marked CHEATING/visu in bss.m and is the actual executed physics-row count'];
report.options = struct( ...
    'use_gp', options.use_gp, ...
    'samplesize', options.samplesize, ...
    'p0', options.p0, ...
    'initial_design_size', options.N0, ...
    'sur_min_eval_per_stage', options.SUR.min_eval_per_stage, ...
    'sur_max_eval_per_stage', options.SUR.max_eval_per_stage);
report.elapsed_seconds = elapsed_seconds;

encoded = jsonencode(report, PrettyPrint=true);
fprintf('\nBSS_ACCEPTANCE_JSON=%s\n', encoded);
if nargin >= 1 && ~isempty(output_json)
    output_json = char(output_json);
    output_parent = fileparts(output_json);
    if ~isempty(output_parent) && ~isfolder(output_parent)
        mkdir(output_parent);
    end
    temporary = [output_json '.tmp'];
    file_id = fopen(temporary, 'w');
    assert(file_id >= 0, 'Cannot open acceptance output: %s', temporary);
    cleanup = onCleanup(@() fclose(file_id)); %#ok<NASGU>
    fprintf(file_id, '%s\n', encoded);
    clear cleanup;
    movefile(temporary, output_json, 'f');
end

assert(strcmp(report.status, 'PASS'), ...
    'BSS author example acceptance failed: Pf ratio %.6g', reference_ratio);

    function values = counted_author_cost_fun(points)
        raw_cost_fun_call_count = raw_cost_fun_call_count + 1;
        raw_cost_fun_row_count = raw_cost_fun_row_count + size(points, 1);
        values = feval(author_cost_fun, points);
    end
end
