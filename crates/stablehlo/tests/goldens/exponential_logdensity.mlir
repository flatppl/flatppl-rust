module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log %arg0 : tensor<f32>
    %6 = stablehlo.multiply %arg0, %4 : tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.add %5, %7 : tensor<f32>
    %9 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.select %2, %8, %10 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %11 : tensor<f32>
  }
}
