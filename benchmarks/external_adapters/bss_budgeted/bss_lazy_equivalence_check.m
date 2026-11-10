function report = bss_lazy_equivalence_check(output_json)
%BSS_LAZY_EQUIVALENCE_CHECK Verify lazy truth lookup is decision-equivalent.
%
% This uses the authors' lightweight official case 006.  It is an adapter
% acceptance check, not a run of any project benchmark case.

expected_commit = '574e083634b9d640c4e6bdfc6b94f923e074ff82';
adapter_root = fileparts(mfilename('fullpath'));
external_root = fileparts(adapter_root);
moretest_root = fileparts(external_root);
bss_root = fullfile(moretest_root, 'third_party', 'bss');
setup_bss_paths(bss_root, expected_commit);

rng(0, 'twister');
base_problem = select_example(6);
options = bss_get_default_options([], base_problem);
options.figs.disable_all = true;
[x0, y0] = bss_initial_design(base_problem);
initial_state = rng;
initial_rows = size(x0, 1);
base_cost_fun = base_problem.cost_fun;
reference_extra_rows = 0;
lazy_extra_rows = 0;

reference_problem = base_problem;
reference_problem.cost_fun = @reference_cost;
rng(initial_state);
reference = bss_instrumented_reference(reference_problem, options, x0, y0);

lazy_problem = base_problem;
lazy_problem.cost_fun = @lazy_cost;
rng(initial_state);
lazy = bss_lazy_oracle(lazy_problem, options, x0, y0);

assert(numel(reference) == numel(lazy), 'Stage-count mismatch.');
fields = {'u_base', 'u_target', 'estim_ratio', 'Pf', 'squaredCoV', ...
    'n_evals', 'selected_x', 'selected_y', 'observation_x', 'observation_y'};
for stage = 1:numel(reference)
    for index = 1:numel(fields)
        field = fields{index};
        assert(isequaln(reference(stage).(field), lazy(stage).(field)), ...
            'BSS:LazyMismatch', 'Mismatch in stage %d field %s.', stage, field);
    end
    if isfield(reference, 'accept_rate')
        assert(isequaln(reference(stage).accept_rate, lazy(stage).accept_rate), ...
            'Acceptance-rate mismatch in stage %d.', stage);
    end
    if isfield(reference, 'sig_RW')
        assert(isequaln(reference(stage).sig_RW, lazy(stage).sig_RW), ...
            'Proposal-scale mismatch in stage %d.', stage);
    end
end

stage_evaluations = [lazy.n_evals];
decision_label_count = initial_rows + sum(stage_evaluations);
lazy_actual_rows = initial_rows + lazy_extra_rows;
reference_actual_rows = initial_rows + reference_extra_rows;
assert(lazy_actual_rows == decision_label_count, ...
    'Lazy physical-row count differs from decision-label count.');
assert(reference_actual_rows > decision_label_count, ...
    'Instrumented author entry did not expose eager pool evaluations.');

report = struct( ...
    'schema', 'bss-lazy-oracle-equivalence-v1', ...
    'status', 'PASS', ...
    'official_case', 6, ...
    'third_party_commit', expected_commit, ...
    'random_seed', 0, ...
    'stages', numel(lazy), ...
    'stage_new_evaluations', stage_evaluations, ...
    'decision_label_count', decision_label_count, ...
    'author_eager_actual_cost_rows', reference_actual_rows, ...
    'lazy_actual_cost_rows', lazy_actual_rows, ...
    'selected_points_identical', true, ...
    'selected_labels_identical', true, ...
    'gp_observation_histories_identical', true, ...
    'thresholds_and_probability_outputs_identical', true, ...
    'smc_mh_outputs_identical', true);
if nargin >= 1 && ~isempty(output_json)
    write_json_atomic(output_json, report);
end
fprintf('BSS_LAZY_EQUIVALENCE=%s\n', jsonencode(report));

    function y = reference_cost(x)
        reference_extra_rows = reference_extra_rows + size(x, 1);
        y = feval(base_cost_fun, x);
    end

    function y = lazy_cost(x)
        lazy_extra_rows = lazy_extra_rows + size(x, 1);
        y = feval(base_cost_fun, x);
    end
end

function setup_bss_paths(bss_root, expected_commit)
assert(isfolder(bss_root), 'Pinned BSS checkout is absent: %s', bss_root);
[status, output] = system(sprintf('git -C "%s" rev-parse HEAD', bss_root));
assert(status == 0 && strcmp(strtrim(output), expected_commit), ...
    'Pinned BSS commit mismatch.');
addpath(fullfile(bss_root, 'scripts'));
addpath(fullfile(bss_root, 'examples'));
addpath(fullfile(bss_root, 'functions'));
addpath(fullfile(bss_root, 'functions', 'distrib'));
addpath(fullfile(bss_root, 'functions', 'bss'));
addpath(fullfile(bss_root, 'stk'));
stk_init;
mex_path = fullfile(bss_root, 'functions', 'bss', ['replicate_particles.' mexext]);
assert(isfile(mex_path), 'Compiled BSS helper is absent: %s', mex_path);
end

function write_json_atomic(path, value)
path = char(string(path));
parent = fileparts(path);
if ~isempty(parent) && ~isfolder(parent), mkdir(parent); end
temporary = [path '.tmp'];
file_id = fopen(temporary, 'w');
assert(file_id >= 0, 'Cannot write %s', temporary);
cleanup = onCleanup(@() fclose(file_id)); %#ok<NASGU>
fprintf(file_id, '%s\n', jsonencode(value, PrettyPrint=true));
clear cleanup;
movefile(temporary, path, 'f');
end
