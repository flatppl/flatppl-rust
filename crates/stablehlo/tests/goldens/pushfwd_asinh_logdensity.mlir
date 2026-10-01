module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = chlo.sinh %arg0 : tensor<f32> -> tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.subtract %0, %1 : tensor<f32>
    %7 = stablehlo.divide %6, %2 : tensor<f32>
    %8 = stablehlo.constant dense<-0.5> : tensor<f32>
    %9 = stablehlo.multiply %7, %7 : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.constant dense<"0x8E3F6BBF"> : tensor<f32>
    %12 = stablehlo.add %11, %10 : tensor<f32>
    %13 = stablehlo.abs %arg0 : tensor<f32>
    %15 = stablehlo.constant dense<"0x000000C0"> : tensor<f32>
    %16 = stablehlo.multiply %15, %13 : tensor<f32>
    %17 = stablehlo.exponential %16 : tensor<f32>
    %18 = stablehlo.log_plus_one %17 : tensor<f32>
    %19 = stablehlo.add %13, %18 : tensor<f32>
    %20 = stablehlo.constant dense<"0x1872313F"> : tensor<f32>
    %21 = stablehlo.subtract %19, %20 : tensor<f32>
    %22 = stablehlo.negate %21 : tensor<f32>
    %23 = stablehlo.subtract %12, %22 : tensor<f32>
    return %23 : tensor<f32>
  }
}
