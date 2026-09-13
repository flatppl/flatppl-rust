module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<-1.0> : tensor<f32>
    %1 = stablehlo.compare GT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.7182817459106445> : tensor<f32>
    %4 = stablehlo.select %1, %arg0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log_plus_one %4 : tensor<f32>
    %6 = stablehlo.constant dense<0.0> : tensor<f32>
    %9 = stablehlo.subtract %5, %6 : tensor<f32>
    %10 = stablehlo.divide %9, %2 : tensor<f32>
    %11 = stablehlo.constant dense<-0.5> : tensor<f32>
    %12 = stablehlo.multiply %10, %10 : tensor<f32>
    %13 = stablehlo.multiply %11, %12 : tensor<f32>
    %14 = stablehlo.constant dense<-0.9189385175704956> : tensor<f32>
    %15 = stablehlo.add %14, %13 : tensor<f32>
    %16 = stablehlo.subtract %15, %5 : tensor<f32>
    %17 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.select %1, %16, %18 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %19 : tensor<f32>
  }
}
