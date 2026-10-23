module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %1 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.constant dense<2.0> : tensor<f32>
    %6 = stablehlo.constant dense<2.5> : tensor<f32>
    %9 = stablehlo.subtract %6, %1 : tensor<f32>
    %10 = stablehlo.divide %9, %3 : tensor<f32>
    %11 = stablehlo.multiply %10, %10 : tensor<f32>
    %12 = stablehlo.multiply %4, %11 : tensor<f32>
    %13 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %14 = stablehlo.multiply %10, %13 : tensor<f32>
    %15 = stablehlo.multiply %4, %14 : tensor<f32>
    %17 = stablehlo.constant dense<"0xFFFFFF3E"> : tensor<f32>
    %18 = stablehlo.add %12, %15 : tensor<f32>
    %19 = stablehlo.add %18, %17 : tensor<f32>
    %20 = stablehlo.constant dense<-0.5> : tensor<f32>
    %21 = stablehlo.multiply %20, %19 : tensor<f32>
    %24 = stablehlo.constant dense<"0x8E3FEB3F"> : tensor<f32>
    %25 = stablehlo.subtract %21, %24 : tensor<f32>
    return %25 : tensor<f32>
  }
}
