#' simulate_AMS
#' Monte Carlo simulation of price paths under:
#'  1 = Black–Scholes (exact solution)
#'  2 = Heston (Euler discretisation)
#'  3 = Heston (Milstein discretisation)
#'  4 = Heston (Quadratic–Exponential scheme, Andersen 2008)
#'
#' @param model Integer in \eqn{\{1,2,3,4\}} selecting the model.
#' @param n Number of simulated paths (>0).
#' @param t Maturity in years (>0).
#' @param p Total time steps (>0).
#' @param r Risk–free rate.
#' @param sigma Black–Scholes volatility (>=0, used only when \code{model == 1}).
#' @param S0 Initial spot price (>0).
#' @param rho Correlation between asset and variance Brownian motions (required for Heston models, finite in \eqn{[-1,1]}).
#' @param rim Left–trim: discard the first \code{rim} time steps (0 <= rim < p). Returned matrices keep \code{p - rim + 1} columns including the initial time.
#' @param v0 Initial variance for Heston models (>=0).
#'
#' @return List: for model 1 returns \code{S} (\eqn{n \times (p-rim+1)}); for Heston models returns \code{S} and \code{V}.
#'
#' @examples
#' \donttest{
#'   b <- simulate_AMS(1, n = 50, t = 1, p = 10, r = 0.01, sigma = 0.2, S0 = 100, rho = NULL)
#'   str(b)
#' }
#' @export
simulate_AMS <- function(model, n, t, p, r, sigma, S0, rho = NULL, rim = 0L, v0 = 0.04) {
  .Call(`_amsSim_simulate_AMS`, model, n, t, p, r, sigma, S0, rho, rim, v0)
}


#' AMS
#' Adaptive Multilevel Splitting estimator for rare-event option payoffs.
#'
#' Pipeline per iteration:
#' \itemize{
#'   \item Simulate \eqn{n} paths under the chosen model (BS/Heston-family).
#'   \item Compute continuation scores \eqn{a_{i,j}} via \code{function_AMS_Cpp}.
#'   \item Set level \eqn{L =} K-th order statistic of \eqn{\max_j a_{i,j}}.
#'   \item Identify survivors (top \eqn{n-K}) and parents (\eqn{K} indices that cleared the level).
#'   \item For each parent, cut at first index that exceeds \eqn{L} and resimulate the suffix.
#'   \item Repeat until \eqn{L \ge L_{\max}}. Then compute discounted payoff on the final population.
#' }
#'
#' @param model 1 = Black–Scholes; 2,3,4 = Heston variants (as in \code{simulate_AMS}).
#' @param type Payoff type passed to \code{payoff()} and \code{function_AMS_Cpp} (1..6).
#' @param funz 1 = BS digital proxy in continuation; 2 = raw feature (signed).
#' @param n Population size (> K).
#' @param t Maturity in years (>0).
#' @param p Total time steps (>0).
#' @param r Risk–free rate.
#' @param sigma BS volatility (used by continuation; >0 if \code{funz == 1}).
#' @param S0 Initial spot.
#' @param rho Correlation for Heston models (required for \code{model >= 2}, in \eqn{[-1,1]}).
#' @param rim Left-trim for simulation (keep last \code{p - rim} steps; \code{0 <= rim < p}).
#' @param v0 Initial variance for Heston models (>=0).
#' @param Lmax Stopping level: iterate while \eqn{L < L_{\max}}.
#' @param strike Strike \eqn{K} used by continuation and final payoff.
#' @param K Number of resampled offspring per iteration (1..n-1).
#'
#' @return List with \code{price} and \code{std}.
#'
#' @examples
#' \donttest{
#'   out <- AMS(model = 2, type = 3, funz = 1, n = 500, t = 1, p = 252, r = 0.03,
#'              sigma = 0.2, rho = -0.5, S0 = 1, rim = 0, Lmax = 0.5, strike = 1.3, K = 200)
#'   str(out)
#' }
#' @export
AMS <- function(model, type, funz, n, t, p, r, sigma, S0, rho = NULL, rim = 0L, v0 = 0.04, Lmax = 0, strike = 1, K = 1L) {
  .Call(`_amsSim_AMS`, model, type, funz, n, t, p, r, sigma, S0, rho, rim, v0, Lmax, strike, K)
}

