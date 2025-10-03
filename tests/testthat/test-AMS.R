test_that("AMS returns price and std with sensible values for funz=1 (BS)", {
  set.seed(3)
  res <- AMS(
    model = 1, type = 1, funz = 1,
    n = 500, t = 1, p = 252,
    r = 0.03, sigma = 0.2, S0 = 1,
    rho = NULL, rim = 0, v0 = 0.04,
    Lmax = 0.5, strike = 1.3, K = 200
  )

  expect_type(res, "list")
  expect_true(all(c("price", "std") %in% names(res)))

  expect_true(is.numeric(res$price) && length(res$price) == 1)
  expect_true(is.numeric(res$std)   && length(res$std)   == 1)

  expect_true(is.finite(res$price))
  expect_true(is.finite(res$std) && res$std >= 0)
})

test_that("AMS returns price and std with sensible values for funz=2 (BS)", {
  set.seed(3)
  res <- AMS(
    model = 1, type = 1, funz = 2,
    n = 500, t = 1, p = 252,
    r = 0.03, sigma = 0.2, S0 = 1,
    rho = NULL, rim = 0, v0 = 0.04,
    Lmax = 1.3, strike = 1.3, K = 200
  )

  expect_type(res, "list")
  expect_true(all(c("price", "std") %in% names(res)))

  expect_true(is.numeric(res$price) && length(res$price) == 1)
  expect_true(is.numeric(res$std)   && length(res$std)   == 1)

  expect_true(is.finite(res$price))
  expect_true(is.finite(res$std) && res$std >= 0)
})

test_that("AMS returns price and std with sensible values for funz=1 (Heston)", {
  set.seed(3)
  res <- AMS(
    model = 2, type = 1, funz = 1,
    n = 500, t = 1, p = 252,
    r = 0.03, sigma = 0.2, S0 = 1,
    rho = -0.5, rim = 0, v0 = 0.04,
    Lmax = 0.5, strike = 1.3, K = 200
  )

  expect_type(res, "list")
  expect_true(all(c("price", "std") %in% names(res)))

  expect_true(is.numeric(res$price) && length(res$price) == 1)
  expect_true(is.numeric(res$std)   && length(res$std)   == 1)

  expect_true(is.finite(res$price))
  expect_true(is.finite(res$std) && res$std >= 0)
})

test_that("AMS returns price and std with sensible values for funz=2 (Heston)", {
  set.seed(3)
  res <- AMS(
    model = 2, type = 1, funz = 2,
    n = 500, t = 1, p = 252,
    r = 0.03, sigma = 0.2, S0 = 1,
    rho = -0.5, rim = 0, v0 = 0.04,
    Lmax =1.3, strike = 1.3, K = 200
  )

  expect_type(res, "list")
  expect_true(all(c("price", "std") %in% names(res)))

  expect_true(is.numeric(res$price) && length(res$price) == 1)
  expect_true(is.numeric(res$std)   && length(res$std)   == 1)

  expect_true(is.finite(res$price))
  expect_true(is.finite(res$std) && res$std >= 0)
})
