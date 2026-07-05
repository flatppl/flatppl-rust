module {
  func.func @logdensity(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.1> : tensor<f32>
    %2 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %5 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %6 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %7 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %8 = stablehlo.compare EQ, %6, %7 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %9 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %10 = stablehlo.select %8, %5, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.log %12 : tensor<2xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.reduce(%13 init: %14) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.multiply %16, %15 : tensor<f32>
    %18 = stablehlo.constant dense<-0.5> : tensor<f32>
    %19 = stablehlo.multiply %18, %17 : tensor<f32>
    %20 = stablehlo.subtract %4, %arg0 : tensor<2xf32>
    %21 = stablehlo.reshape %20 : (tensor<2xf32>) -> tensor<2x1xf32>
    %22 = "stablehlo.triangular_solve"(%5, %21) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<2x1xf32>) -> tensor<2xf32>
    %24 = stablehlo.multiply %23, %23 : tensor<2xf32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.reduce(%24 init: %25) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %27 = stablehlo.multiply %18, %26 : tensor<f32>
    %28 = stablehlo.constant dense<-1.8378770664093453> : tensor<f32>
    %29 = stablehlo.add %28, %19 : tensor<f32>
    %30 = stablehlo.add %29, %27 : tensor<f32>
    return %30 : tensor<f32>
  }
}
