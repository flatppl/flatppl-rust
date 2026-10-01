module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.multiply %0, %arg0 : tensor<f32>
    %5 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %6 = stablehlo.multiply %4, %5 : tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = chlo.lgamma %4 : tensor<f32> -> tensor<f32>
    %9 = stablehlo.negate %8 : tensor<f32>
    %10 = stablehlo.subtract %4, %3 : tensor<f32>
    %11 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %12 = stablehlo.multiply %10, %11 : tensor<f32>
    %15 = stablehlo.constant dense<"0x000080BE"> : tensor<f32>
    %16 = stablehlo.add %7, %9 : tensor<f32>
    %17 = stablehlo.add %16, %12 : tensor<f32>
    %18 = stablehlo.add %17, %15 : tensor<f32>
    return %18 : tensor<f32>
  }
}
