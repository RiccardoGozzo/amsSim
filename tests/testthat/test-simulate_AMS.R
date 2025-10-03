library(testthat)
library(amsSim)

test_that("simulate_AMS works for Black–Scholes (model = 1)", {
  set.seed(1)
  n   <- 100
  p   <- 252
  rim<-0
  out <- simulate_AMS(
    model = 1, n = n, t = 1, p = p,
    r = 0.03, sigma = 0.2, S0 = 100,
    rho = NULL)

  # Expect a list with S
  expect_type(out, "list")
  expect_true("S" %in% names(out))

  # Dimensions: n x (p - rim + 1)
  expect_equal(dim(out$S), c(n, p - rim + 1))

  # Positive prices
  expect_true(all(out$S > 0))
})


test_that("simulate_AMS works for Heston (model = 2)", {
  set.seed(2)
  n   <- 100
  p   <- 252
  rim<-0
  out <- simulate_AMS(
    model = 2, n = n, t = 0.2, p = p,
    r = 0.03, sigma = 0.2, S0 = 100,
    rho = -0.5)

  # Expect a list with S and V
  expect_type(out, "list")
  expect_true(all(c("S", "V") %in% names(out)))

  # Consistent shapes
  expect_equal(dim(out$S), c(n, p - rim + 1))
  expect_equal(dim(out$V), c(n, p - rim + 1))

  # Sanity checks
  expect_true(all(is.finite(out$S)))
  expect_true(all(out$S > 0))
  expect_true(all(is.finite(out$V)))
  expect_true(all(out$V >= 0))
})

test_that("simulate_AMS works for Heston (model = 3)", {
  set.seed(2)
  n   <- 100
  p   <- 252
  rim<-0
  out <- simulate_AMS(
    model = 3, n = n, t = 0.2, p = p,
    r = 0.03, sigma = 0.2, S0 = 100,
    rho = -0.5)

  # Expect a list with S and V
  expect_type(out, "list")
  expect_true(all(c("S", "V") %in% names(out)))

  # Consistent shapes
  expect_equal(dim(out$S), c(n, p - rim + 1))
  expect_equal(dim(out$V), c(n, p - rim + 1))

  # Sanity checks
  expect_true(all(is.finite(out$S)))
  expect_true(all(out$S > 0))
  expect_true(all(is.finite(out$V)))
  expect_true(all(out$V >= 0))
})

test_that("simulate_AMS works for Heston (model = 4)", {
  set.seed(2)
  n   <- 100
  p   <- 252
  rim<-0
  out <- simulate_AMS(
    model = 4, n = n, t = 0.2, p = p,
    r = 0.03, sigma = 0.2, S0 = 100,
    rho = -0.5)

  # Expect a list with S and V
  expect_type(out, "list")
  expect_true(all(c("S", "V") %in% names(out)))

  # Consistent shapes
  expect_equal(dim(out$S), c(n, p - rim + 1))
  expect_equal(dim(out$V), c(n, p - rim + 1))

  # Sanity checks
  expect_true(all(is.finite(out$S)))
  expect_true(all(out$S > 0))
  expect_true(all(is.finite(out$V)))
  expect_true(all(out$V >= 0))
})
