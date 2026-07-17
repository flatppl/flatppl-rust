module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.compare EQ, %1, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %6 = stablehlo.select %2, %3, %5 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }
}
