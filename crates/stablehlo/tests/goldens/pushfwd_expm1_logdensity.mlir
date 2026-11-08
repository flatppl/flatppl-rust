module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<-1.0> : tensor<f32>
    %1 = stablehlo.compare GT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<"0xA8F0DB3F"> : tensor<f32>
    %4 = stablehlo.select %1, %arg0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log_plus_one %4 : tensor<f32>
    %6 = stablehlo.constant dense<0.0> : tensor<f32>
    %10 = stablehlo.subtract %5, %6 : tensor<f32>
    %11 = stablehlo.divide %10, %2 : tensor<f32>
    %12 = stablehlo.constant dense<-0.5> : tensor<f32>
    %13 = stablehlo.multiply %11, %11 : tensor<f32>
    %14 = stablehlo.multiply %12, %13 : tensor<f32>
    %15 = stablehlo.constant dense<"0x8E3F6BBF"> : tensor<f32>
    %16 = stablehlo.add %15, %14 : tensor<f32>
    %17 = stablehlo.abs %5 : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %19 = stablehlo.compare EQ, %17, %18 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %20 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %21 = stablehlo.select %19, %20, %16 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %22 = stablehlo.subtract %21, %5 : tensor<f32>
    %23 = stablehlo.negate %18 : tensor<f32>
    %24 = stablehlo.select %1, %22, %23 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %24 : tensor<f32>
  }
}
