module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare GT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.compare LT, %arg0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.and %1, %3 : tensor<i1>
    %5 = stablehlo.logistic %2 : tensor<f32>
    %6 = stablehlo.select %4, %arg0, %5 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.subtract %2, %6 : tensor<f32>
    %8 = stablehlo.divide %6, %7 : tensor<f32>
    %9 = stablehlo.log %8 : tensor<f32>
    %12 = stablehlo.subtract %9, %0 : tensor<f32>
    %13 = stablehlo.divide %12, %2 : tensor<f32>
    %14 = stablehlo.constant dense<-0.5> : tensor<f32>
    %15 = stablehlo.multiply %13, %13 : tensor<f32>
    %16 = stablehlo.multiply %14, %15 : tensor<f32>
    %17 = stablehlo.constant dense<-0.9189385175704956> : tensor<f32>
    %18 = stablehlo.add %17, %16 : tensor<f32>
    %19 = stablehlo.log %6 : tensor<f32>
    %20 = stablehlo.negate %6 : tensor<f32>
    %21 = stablehlo.log_plus_one %20 : tensor<f32>
    %22 = stablehlo.add %19, %21 : tensor<f32>
    %23 = stablehlo.subtract %18, %22 : tensor<f32>
    %24 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %25 = stablehlo.negate %24 : tensor<f32>
    %26 = stablehlo.select %4, %23, %25 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %26 : tensor<f32>
  }
}
