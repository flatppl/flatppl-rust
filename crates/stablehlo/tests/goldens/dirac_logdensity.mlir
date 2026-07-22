module {
  func.func @logdensity(%arg0: tensor<i32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.compare EQ, %0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %4 = stablehlo.negate %3 : tensor<f32>
    %5 = stablehlo.select %1, %2, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %5 : tensor<f32>
  }
}
