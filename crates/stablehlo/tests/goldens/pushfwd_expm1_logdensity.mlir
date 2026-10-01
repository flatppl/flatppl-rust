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
    %17 = stablehlo.subtract %16, %5 : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.select %1, %17, %19 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %20 : tensor<f32>
  }
}
