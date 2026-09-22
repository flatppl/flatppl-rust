module {
  func.func @logdensity(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<[0.2, 0.1]> : tensor<2xf32>
    %1 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %2 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %3 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %4 = stablehlo.compare EQ, %2, %3 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %5 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %6 = stablehlo.select %4, %1, %5 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %9 = stablehlo.log %8 : tensor<2xf32>
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%9 init: %10) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.constant dense<2.0> : tensor<f32>
    %13 = stablehlo.multiply %12, %11 : tensor<f32>
    %14 = stablehlo.constant dense<-0.5> : tensor<f32>
    %15 = stablehlo.multiply %14, %13 : tensor<f32>
    %16 = stablehlo.subtract %0, %arg0 : tensor<2xf32>
    %17 = stablehlo.reshape %16 : (tensor<2xf32>) -> tensor<2x1xf32>
    %18 = "stablehlo.triangular_solve"(%1, %17) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    %19 = stablehlo.reshape %18 : (tensor<2x1xf32>) -> tensor<2xf32>
    %20 = stablehlo.multiply %19, %19 : tensor<2xf32>
    %21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%20 init: %21) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.multiply %14, %22 : tensor<f32>
    %24 = stablehlo.constant dense<-1.8378770664093453> : tensor<f32>
    %25 = stablehlo.add %24, %15 : tensor<f32>
    %26 = stablehlo.add %25, %23 : tensor<f32>
    return %26 : tensor<f32>
  }
}
