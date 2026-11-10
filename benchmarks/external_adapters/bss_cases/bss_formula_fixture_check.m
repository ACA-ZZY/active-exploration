function report = bss_formula_fixture_check(output_json)
%BSS_FORMULA_FIXTURE_CHECK Compare MATLAB formulas with saved Python labels.

adapter_root = fileparts(mfilename('fullpath'));
fixture_path = fullfile(adapter_root, 'formula_fixture.json');
fixture = jsondecode(fileread(fixture_path));
points = fixture.points;
errors = zeros(numel(points), 1);
for index = 1:numel(points)
    observed = bss_project_response(points(index).case, points(index).x(:)');
    errors(index) = abs(observed - points(index).python_g);
end
assert(max(errors) <= 1e-11, ...
    'BSS:FormulaDrift', 'Project formula mismatch: max error %.17g', max(errors));

report = struct( ...
    'schema', 'bss-project-formula-check-v1', ...
    'status', 'PASS', ...
    'fixture', fixture_path, ...
    'number_of_saved_points', numel(points), ...
    'max_absolute_error', max(errors), ...
    'no_new_training', true, ...
    'no_reference_passed_to_bss', true);

if nargin >= 1 && ~isempty(output_json)
    write_json_atomic(output_json, report);
end
fprintf('BSS_FORMULA_CHECK=%s\n', jsonencode(report));
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
