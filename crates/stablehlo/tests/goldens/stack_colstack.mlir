module {
  func.func @logdensity() -> tensor<3x2xf32> {
    %2 = stablehlo.constant dense<"0x0000803F0000004000004040000080400000A0400000C040"> : tensor<2x3xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %3 : tensor<3x2xf32>
  }
}
