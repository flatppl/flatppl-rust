module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %2 = stablehlo.compare LE, %arg0, %1 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %3 = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%2 init: %3) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<9> : tensor<i32>
    %7 = stablehlo.select %4, %6, %5 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%arg0 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %11 = chlo.lgamma %10 : tensor<f32> -> tensor<f32>
    %12 = chlo.lgamma %arg0 : tensor<3xf32> -> tensor<3xf32>
    %13 = stablehlo.reduce(%12 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %16 = stablehlo.subtract %arg0, %15 : tensor<3xf32>
    %17 = stablehlo.constant dense<"0x1002CEBFC81B9ABF187231BF"> : tensor<3xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<3xf32>
    %19 = stablehlo.reduce(%18 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.add %11, %14 : tensor<f32>
    %21 = stablehlo.add %20, %19 : tensor<f32>
    return %21, %7 : tensor<f32>, tensor<i32>
  }
}
