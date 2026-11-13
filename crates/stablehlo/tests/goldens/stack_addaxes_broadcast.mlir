module {
  func.func @logdensity() -> tensor<3x2xf32> {
    %8 = stablehlo.constant dense<"0x0000803F000020410000804000002042000010410000B442"> : tensor<3x2xf32>
    return %8 : tensor<3x2xf32>
  }
}
