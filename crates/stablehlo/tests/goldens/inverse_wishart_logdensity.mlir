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
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%9 init: %10) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.constant dense<2.0> : tensor<f32>
    %13 = stablehlo.multiply %12, %11 : tensor<f32>
    %14 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %15 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %16 = stablehlo.compare EQ, %14, %15 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %17 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %18 = stablehlo.select %16, %1, %17 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.reduce(%18 init: %19) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %21 = stablehlo.log %20 : tensor<2xf32>
    %22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %23 = stablehlo.reduce(%21 init: %22) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.constant dense<2.0> : tensor<f32>
    %25 = stablehlo.multiply %24, %23 : tensor<f32>
    %26 = "stablehlo.triangular_solve"(%1, %0) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %27 = stablehlo.multiply %26, %26 : tensor<2x2xf32>
    %28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %29 = stablehlo.reduce(%27 init: %28) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = stablehlo.reduce(%29 init: %30) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.constant dense<0.5> : tensor<f32>
    %33 = stablehlo.multiply %32, %arg1 : tensor<f32>
    %34 = stablehlo.multiply %33, %13 : tensor<f32>
    %35 = stablehlo.constant dense<3.0> : tensor<f32>
    %36 = stablehlo.add %arg1, %35 : tensor<f32>
    %37 = stablehlo.constant dense<-0.5> : tensor<f32>
    %38 = stablehlo.multiply %37, %36 : tensor<f32>
    %39 = stablehlo.multiply %38, %25 : tensor<f32>
    %40 = stablehlo.multiply %37, %31 : tensor<f32>
    %41 = stablehlo.constant dense<2.0> : tensor<f32>
    %42 = stablehlo.multiply %arg1, %41 : tensor<f32>
    %43 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %44 = stablehlo.multiply %42, %43 : tensor<f32>
    %45 = stablehlo.multiply %37, %44 : tensor<f32>
    %46 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %47 = stablehlo.constant dense<0.0> : tensor<f32>
    %48 = stablehlo.add %33, %47 : tensor<f32>
    %49 = chlo.lgamma %48 : tensor<f32> -> tensor<f32>
    %50 = stablehlo.add %46, %49 : tensor<f32>
    %51 = stablehlo.constant dense<-0.5> : tensor<f32>
    %52 = stablehlo.add %33, %51 : tensor<f32>
    %53 = chlo.lgamma %52 : tensor<f32> -> tensor<f32>
    %54 = stablehlo.add %50, %53 : tensor<f32>
    %55 = stablehlo.negate %54 : tensor<f32>
    %56 = stablehlo.add %34, %39 : tensor<f32>
    %57 = stablehlo.add %56, %40 : tensor<f32>
    %58 = stablehlo.add %57, %45 : tensor<f32>
    %59 = stablehlo.add %58, %55 : tensor<f32>
    return %59 : tensor<f32>
  }
}
