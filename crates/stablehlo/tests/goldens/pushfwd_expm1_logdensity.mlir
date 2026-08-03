module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<-1.0> : tensor<f32>
    %1 = stablehlo.compare GT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.exponential_minus_one %2 : tensor<f32>
    %4 = stablehlo.select %1, %arg0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log_plus_one %4 : tensor<f32>
    %6 = stablehlo.constant dense<0.0> : tensor<f32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.log %7 : tensor<f32>
    %9 = stablehlo.negate %8 : tensor<f32>
    %10 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %11 = stablehlo.subtract %5, %6 : tensor<f32>
    %12 = stablehlo.divide %11, %7 : tensor<f32>
    %13 = stablehlo.constant dense<-0.5> : tensor<f32>
    %14 = stablehlo.multiply %12, %12 : tensor<f32>
    %15 = stablehlo.multiply %13, %14 : tensor<f32>
    %16 = stablehlo.add %9, %10 : tensor<f32>
    %17 = stablehlo.add %16, %15 : tensor<f32>
    %18 = stablehlo.log_plus_one %4 : tensor<f32>
    %19 = stablehlo.subtract %17, %18 : tensor<f32>
    %20 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.select %1, %19, %21 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %22 : tensor<f32>
  }
}
