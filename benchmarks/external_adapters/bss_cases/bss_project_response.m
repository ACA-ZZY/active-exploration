function g = bss_project_response(case_slug, x)
%BSS_PROJECT_RESPONSE Exact project response g(x) for the four supported BSS cases.
%
% X is always in the project's latent standard-normal coordinates.  This
% function deliberately contains no probability or reference-threshold data.

case_slug = char(string(case_slug));
x = double(x);
if isempty(x) || ~ismatrix(x) || any(~isfinite(x), 'all')
    error('BSS:InvalidInput', 'x must be a finite non-empty matrix.');
end

switch case_slug
    case 'two-sided-kink'
        assert(size(x, 2) == 2, 'two-sided-kink requires two columns.');
        beta = 5.05;
        kappa = 1.0;
        s0 = 3.4;
        s = abs(sum(x, 2) ./ sqrt(2.0));
        g = beta - s - kappa .* max(s - s0, 0.0);

    case 'radial-kink'
        assert(size(x, 2) == 8, 'radial-kink requires eight columns.');
        beta = 8.7;
        kappa = 1.0;
        q0 = 4.3;
        q = (sum(x .* x, 2) - 8.0) ./ 4.0;
        g = beta - q - kappa .* max(q - q0, 0.0);

    case 'tail-kink'
        assert(size(x, 2) == 50, 'tail-kink requires fifty columns.');
        s = sum(x, 2) ./ sqrt(50.0);
        g = 4.8 - s - max(s - 3.2, 0.0);

    case 'cantilever_beam_2d'
        assert(size(x, 2) == 2, 'cantilever_beam_2d requires two columns.');
        w_pa = 1000.0 + 200.0 .* x(:, 1);
        h_mm = 250.0 + 37.5 .* x(:, 2);
        modulus_pa = 2.6e4 * 1.0e6;
        span_mm = 6.0e3;
        g = span_mm / 325.0 - (12.0 / 8.0) * span_mm^4 / modulus_pa ...
            .* w_pa ./ (h_mm .^ 3);

    otherwise
        error('BSS:UnknownCase', 'Unknown BSS project case: %s', case_slug);
end

g = reshape(double(g), [], 1);
end
