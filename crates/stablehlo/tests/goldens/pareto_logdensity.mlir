module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.compare GE, %0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.select %1, %0, %arg1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = stablehlo.log %arg0 : tensor<f32>
    %4 = stablehlo.log %arg1 : tensor<f32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<f32>
    %6 = stablehlo.constant dense<1.0> : tensor<f32>
    %7 = stablehlo.add %arg0, %6 : tensor<f32>
    %8 = stablehlo.log %2 : tensor<f32>
    %9 = stablehlo.multiply %7, %8 : tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.add %3, %5 : tensor<f32>
    %12 = stablehlo.add %11, %10 : tensor<f32>
    %13 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.select %1, %12, %14 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %15 : tensor<f32>
  }
}
