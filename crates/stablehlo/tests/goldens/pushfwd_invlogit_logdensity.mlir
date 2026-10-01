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
    %13 = stablehlo.subtract %9, %0 : tensor<f32>
    %14 = stablehlo.divide %13, %2 : tensor<f32>
    %15 = stablehlo.constant dense<-0.5> : tensor<f32>
    %16 = stablehlo.multiply %14, %14 : tensor<f32>
    %17 = stablehlo.multiply %15, %16 : tensor<f32>
    %18 = stablehlo.constant dense<"0x8E3F6BBF"> : tensor<f32>
    %19 = stablehlo.add %18, %17 : tensor<f32>
    %20 = stablehlo.log %6 : tensor<f32>
    %21 = stablehlo.negate %6 : tensor<f32>
    %22 = stablehlo.log_plus_one %21 : tensor<f32>
    %23 = stablehlo.add %20, %22 : tensor<f32>
    %24 = stablehlo.subtract %19, %23 : tensor<f32>
    %25 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %26 = stablehlo.negate %25 : tensor<f32>
    %27 = stablehlo.select %4, %24, %26 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %27 : tensor<f32>
  }
}
