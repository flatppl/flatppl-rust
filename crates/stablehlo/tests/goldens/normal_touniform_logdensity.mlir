module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %3 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %4 = stablehlo.constant dense<1.4142135623730951> : tensor<f32>
    %5 = stablehlo.multiply %arg1, %4 : tensor<f32>
    %6 = stablehlo.divide %3, %5 : tensor<f32>
    %7 = chlo.erf %6 : tensor<f32> -> tensor<f32>
    %8 = stablehlo.add %0, %7 : tensor<f32>
    %9 = stablehlo.constant dense<0.5> : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.subtract %0, %10 : tensor<f32>
    %12 = stablehlo.constant dense<0.0> : tensor<f32>
    %13 = stablehlo.compare EQ, %11, %12 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %14 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %15 = stablehlo.select %13, %0, %11 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.log %15 : tensor<f32>
    %17 = stablehlo.select %13, %14, %16 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.abs %17 : tensor<f32>
    %19 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %20 = stablehlo.compare LT, %18, %19 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %21 = stablehlo.constant dense<true> : tensor<i1>
    %22 = stablehlo.compare LE, %9, %19 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %23 = stablehlo.and %21, %22 : tensor<i1>
    %24 = stablehlo.select %23, %9, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.log %arg1 : tensor<f32>
    %26 = stablehlo.negate %25 : tensor<f32>
    %27 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %28 = stablehlo.subtract %24, %arg0 : tensor<f32>
    %29 = stablehlo.divide %28, %arg1 : tensor<f32>
    %30 = stablehlo.constant dense<-0.5> : tensor<f32>
    %31 = stablehlo.multiply %29, %29 : tensor<f32>
    %32 = stablehlo.multiply %30, %31 : tensor<f32>
    %33 = stablehlo.add %26, %27 : tensor<f32>
    %34 = stablehlo.add %33, %32 : tensor<f32>
    %35 = stablehlo.negate %19 : tensor<f32>
    %36 = stablehlo.select %23, %34, %35 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %37 = stablehlo.subtract %36, %17 : tensor<f32>
    %38 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %39 = stablehlo.select %20, %37, %38 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %39 : tensor<f32>
  }
}
