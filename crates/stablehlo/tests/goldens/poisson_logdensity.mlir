module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<"0x00004040"> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare EQ, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.select %5, %4, %arg0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.log %6 : tensor<f32>
    %11 = stablehlo.multiply %2, %10 : tensor<f32>
    %13 = stablehlo.constant dense<"0x6058E53F"> : tensor<f32>
    %14 = stablehlo.subtract %11, %6 : tensor<f32>
    %15 = stablehlo.subtract %14, %13 : tensor<f32>
    %60 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %61 = stablehlo.negate %60 : tensor<f32>
    %62 = stablehlo.select %5, %61, %15 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %62 : tensor<f32>
  }
}
