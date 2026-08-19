module {
  func.func @logdensity(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
    %0 = stablehlo.slice %arg0 [0:2, 0:1] : (tensor<2x3xf32>) -> tensor<2x1xf32>
    %1 = stablehlo.reshape %0 : (tensor<2x1xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
