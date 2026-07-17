module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log %4 : tensor<f32>
    %6 = stablehlo.negate %5 : tensor<f32>
    %7 = stablehlo.log %arg1 : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %10 = stablehlo.subtract %5, %arg0 : tensor<f32>
    %11 = stablehlo.divide %10, %arg1 : tensor<f32>
    %12 = stablehlo.constant dense<-0.5> : tensor<f32>
    %13 = stablehlo.multiply %11, %11 : tensor<f32>
    %14 = stablehlo.multiply %12, %13 : tensor<f32>
    %15 = stablehlo.add %6, %8 : tensor<f32>
    %16 = stablehlo.add %15, %9 : tensor<f32>
    %17 = stablehlo.add %16, %14 : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.select %2, %17, %19 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %20 : tensor<f32>
  }
}
