module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.compare GE, %0, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.compare LE, %0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.and %4, %5 : tensor<i1>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.select %6, %0, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.log %arg1 : tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %12 = stablehlo.subtract %8, %arg0 : tensor<f32>
    %13 = stablehlo.divide %12, %arg1 : tensor<f32>
    %14 = stablehlo.constant dense<-0.5> : tensor<f32>
    %15 = stablehlo.multiply %13, %13 : tensor<f32>
    %16 = stablehlo.multiply %14, %15 : tensor<f32>
    %17 = stablehlo.add %10, %11 : tensor<f32>
    %18 = stablehlo.add %17, %16 : tensor<f32>
    %19 = stablehlo.negate %2 : tensor<f32>
    %20 = stablehlo.select %6, %18, %19 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %21 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %22 = stablehlo.constant dense<1.4142135623730951> : tensor<f32>
    %23 = stablehlo.multiply %arg1, %22 : tensor<f32>
    %24 = stablehlo.divide %21, %23 : tensor<f32>
    %25 = chlo.erf %24 : tensor<f32> -> tensor<f32>
    %26 = stablehlo.add %7, %25 : tensor<f32>
    %27 = stablehlo.multiply %0, %26 : tensor<f32>
    %28 = stablehlo.subtract %3, %arg0 : tensor<f32>
    %29 = stablehlo.divide %28, %23 : tensor<f32>
    %30 = chlo.erf %29 : tensor<f32> -> tensor<f32>
    %31 = stablehlo.add %7, %30 : tensor<f32>
    %32 = stablehlo.multiply %0, %31 : tensor<f32>
    %33 = stablehlo.subtract %27, %32 : tensor<f32>
    %34 = stablehlo.log %33 : tensor<f32>
    %35 = stablehlo.subtract %20, %34 : tensor<f32>
    return %35 : tensor<f32>
  }
}
