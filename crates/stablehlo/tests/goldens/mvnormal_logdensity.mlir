module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<f32>, tensor<i32>) {
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
    %11 = stablehlo.subtract %10, %10 : tensor<f32>
    %12 = stablehlo.compare EQ, %11, %11 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %13 = stablehlo.constant dense<0> : tensor<i32>
    %14 = stablehlo.constant dense<9> : tensor<i32>
    %15 = stablehlo.select %12, %13, %14 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.multiply %16, %10 : tensor<f32>
    %18 = stablehlo.constant dense<-0.5> : tensor<f32>
    %19 = stablehlo.multiply %18, %17 : tensor<f32>
    %20 = stablehlo.subtract %0, %arg0 : tensor<2xf32>
    %21 = stablehlo.reshape %20 : (tensor<2xf32>) -> tensor<1x2xf32>
    %22 = "stablehlo.triangular_solve"(%1, %21) <{left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>}> : (tensor<2x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %23 = stablehlo.reshape %22 : (tensor<1x2xf32>) -> tensor<2xf32>
    %24 = stablehlo.multiply %23, %23 : tensor<2xf32>
    %25 = stablehlo.reduce(%24 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %26 = stablehlo.multiply %18, %25 : tensor<f32>
    %27 = stablehlo.constant dense<-1.8378770664093453> : tensor<f32>
    %28 = stablehlo.add %27, %19 : tensor<f32>
    %29 = stablehlo.add %28, %26 : tensor<f32>
    return %29, %15 : tensor<f32>, tensor<i32>
  }
}
