module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare GT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.compare LT, %arg0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.and %1, %3 : tensor<i1>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.logistic %5 : tensor<f32>
    %7 = stablehlo.select %4, %arg0, %6 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.subtract %8, %7 : tensor<f32>
    %10 = stablehlo.divide %7, %9 : tensor<f32>
    %11 = stablehlo.log %10 : tensor<f32>
    %12 = stablehlo.constant dense<0.0> : tensor<f32>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.log %13 : tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %17 = stablehlo.subtract %11, %12 : tensor<f32>
    %18 = stablehlo.divide %17, %13 : tensor<f32>
    %19 = stablehlo.constant dense<-0.5> : tensor<f32>
    %20 = stablehlo.multiply %18, %18 : tensor<f32>
    %21 = stablehlo.multiply %19, %20 : tensor<f32>
    %22 = stablehlo.add %15, %16 : tensor<f32>
    %23 = stablehlo.add %22, %21 : tensor<f32>
    %24 = stablehlo.log %7 : tensor<f32>
    %25 = stablehlo.negate %7 : tensor<f32>
    %26 = stablehlo.log_plus_one %25 : tensor<f32>
    %27 = stablehlo.add %24, %26 : tensor<f32>
    %28 = stablehlo.subtract %23, %27 : tensor<f32>
    %29 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %30 = stablehlo.negate %29 : tensor<f32>
    %31 = stablehlo.select %4, %28, %30 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %31 : tensor<f32>
  }
}
