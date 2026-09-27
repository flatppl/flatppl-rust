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
    %10 = stablehlo.reduce(%9 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.constant dense<2.0> : tensor<f32>
    %12 = stablehlo.multiply %11, %10 : tensor<f32>
    %13 = stablehlo.constant dense<-0.5> : tensor<f32>
    %14 = stablehlo.multiply %13, %12 : tensor<f32>
    %15 = stablehlo.subtract %0, %arg0 : tensor<2xf32>
    %16 = stablehlo.reshape %15 : (tensor<2xf32>) -> tensor<2x1xf32>
    %17 = "stablehlo.triangular_solve"(%1, %16) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    %18 = stablehlo.reshape %17 : (tensor<2x1xf32>) -> tensor<2xf32>
    %19 = stablehlo.multiply %18, %18 : tensor<2xf32>
    %20 = stablehlo.reduce(%19 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %21 = stablehlo.multiply %13, %20 : tensor<f32>
    %22 = stablehlo.constant dense<-1.8378770664093453> : tensor<f32>
    %23 = stablehlo.add %22, %14 : tensor<f32>
    %24 = stablehlo.add %23, %21 : tensor<f32>
    return %24 : tensor<f32>
  }
}
