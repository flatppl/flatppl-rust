module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %7 = stablehlo.constant dense<2.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %19 = stablehlo.constant dense<"0x00002040"> : tensor<f32>
    %23 = stablehlo.subtract %19, %arg0 : tensor<f32>
    %24 = stablehlo.divide %23, %8 : tensor<f32>
    %25 = stablehlo.constant dense<-0.5> : tensor<f32>
    %26 = stablehlo.multiply %24, %24 : tensor<f32>
    %27 = stablehlo.multiply %25, %26 : tensor<f32>
    %28 = stablehlo.constant dense<"0x8E3F6BBF"> : tensor<f32>
    %29 = stablehlo.add %28, %27 : tensor<f32>
    %30 = stablehlo.multiply %7, %29 : tensor<f32>
    %45 = stablehlo.constant dense<"0xFFFF7F3E"> : tensor<f32>
    %46 = stablehlo.subtract %30, %45 : tensor<f32>
    return %46 : tensor<f32>
  }
}
