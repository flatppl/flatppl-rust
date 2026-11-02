module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.compare LE, %1, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%arg0 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %10 = chlo.lgamma %9 : tensor<f32> -> tensor<f32>
    %11 = chlo.lgamma %arg0 : tensor<3xf32> -> tensor<3xf32>
    %12 = stablehlo.reduce(%11 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %15 = stablehlo.subtract %arg0, %14 : tensor<3xf32>
    %16 = stablehlo.constant dense<"0x1002CEBFC81B9ABF187231BF"> : tensor<3xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<3xf32>
    %18 = stablehlo.reduce(%17 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %19 = stablehlo.add %10, %13 : tensor<f32>
    %20 = stablehlo.add %19, %18 : tensor<f32>
    return %20, %6 : tensor<f32>, tensor<i32>
  }
}
