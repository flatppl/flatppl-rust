module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<3.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare EQ, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.select %5, %4, %arg0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.log %6 : tensor<f32>
    %11 = stablehlo.multiply %2, %10 : tensor<f32>
    %13 = stablehlo.constant dense<1.7917594909667969> : tensor<f32>
    %14 = stablehlo.subtract %11, %6 : tensor<f32>
    %15 = stablehlo.subtract %14, %13 : tensor<f32>
    %57 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %58 = stablehlo.negate %57 : tensor<f32>
    %59 = stablehlo.select %5, %58, %15 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %59 : tensor<f32>
  }
}
