module {
  func.func @logdensity(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2x2xf32>) -> tensor<f32> {
    %0 = stablehlo.cholesky %arg0, lower = true : tensor<2x2xf32>
    %1 = stablehlo.cholesky %arg2, lower = true : tensor<2x2xf32>
    %2 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %3 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %4 = stablehlo.compare EQ, %2, %3 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %5 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %6 = stablehlo.select %4, %0, %5 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %9 = stablehlo.log %8 : tensor<2xf32>
    %10 = stablehlo.reduce(%9 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.constant dense<2.0> : tensor<f32>
    %12 = stablehlo.multiply %11, %10 : tensor<f32>
    %13 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %14 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %15 = stablehlo.compare EQ, %13, %14 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %16 = stablehlo.select %15, %1, %5 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %17 = stablehlo.reduce(%16 init: %7) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %18 = stablehlo.log %17 : tensor<2xf32>
    %19 = stablehlo.reduce(%18 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.multiply %11, %19 : tensor<f32>
    %21 = "stablehlo.triangular_solve"(%1, %0) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %22 = stablehlo.multiply %21, %21 : tensor<2x2xf32>
    %23 = stablehlo.reduce(%22 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %24 = stablehlo.reduce(%23 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.constant dense<0.5> : tensor<f32>
    %26 = stablehlo.multiply %25, %arg1 : tensor<f32>
    %27 = stablehlo.multiply %26, %12 : tensor<f32>
    %28 = stablehlo.constant dense<3.0> : tensor<f32>
    %29 = stablehlo.add %arg1, %28 : tensor<f32>
    %30 = stablehlo.constant dense<-0.5> : tensor<f32>
    %31 = stablehlo.multiply %30, %29 : tensor<f32>
    %32 = stablehlo.multiply %31, %20 : tensor<f32>
    %33 = stablehlo.multiply %30, %24 : tensor<f32>
    %34 = stablehlo.multiply %arg1, %11 : tensor<f32>
    %35 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %36 = stablehlo.multiply %34, %35 : tensor<f32>
    %37 = stablehlo.multiply %30, %36 : tensor<f32>
    %38 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %39 = stablehlo.constant dense<0.0> : tensor<f32>
    %40 = stablehlo.add %26, %39 : tensor<f32>
    %41 = chlo.lgamma %40 : tensor<f32> -> tensor<f32>
    %42 = stablehlo.add %38, %41 : tensor<f32>
    %43 = stablehlo.add %26, %30 : tensor<f32>
    %44 = chlo.lgamma %43 : tensor<f32> -> tensor<f32>
    %45 = stablehlo.add %42, %44 : tensor<f32>
    %46 = stablehlo.negate %45 : tensor<f32>
    %47 = stablehlo.add %27, %32 : tensor<f32>
    %48 = stablehlo.add %47, %33 : tensor<f32>
    %49 = stablehlo.add %48, %37 : tensor<f32>
    %50 = stablehlo.add %49, %46 : tensor<f32>
    return %50 : tensor<f32>
  }
}
