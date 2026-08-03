module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %4 = stablehlo.subtract %0, %3 : tensor<f32>
    %5 = stablehlo.subtract %2, %0 : tensor<f32>
    %6 = stablehlo.multiply %4, %5 : tensor<f32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.compare GE, %6, %7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.constant dense<1.0> : tensor<f32>
    %10 = stablehlo.select %8, %0, %9 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.log %arg1 : tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %14 = stablehlo.subtract %10, %arg0 : tensor<f32>
    %15 = stablehlo.divide %14, %arg1 : tensor<f32>
    %16 = stablehlo.constant dense<-0.5> : tensor<f32>
    %17 = stablehlo.multiply %15, %15 : tensor<f32>
    %18 = stablehlo.multiply %16, %17 : tensor<f32>
    %19 = stablehlo.add %12, %13 : tensor<f32>
    %20 = stablehlo.add %19, %18 : tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %22 = stablehlo.negate %21 : tensor<f32>
    %23 = stablehlo.select %8, %20, %22 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %25 = stablehlo.constant dense<1.4142135623730951> : tensor<f32>
    %26 = stablehlo.multiply %arg1, %25 : tensor<f32>
    %27 = stablehlo.divide %24, %26 : tensor<f32>
    %28 = chlo.erf %27 : tensor<f32> -> tensor<f32>
    %29 = stablehlo.constant dense<1.0> : tensor<f32>
    %30 = stablehlo.add %29, %28 : tensor<f32>
    %31 = stablehlo.constant dense<0.5> : tensor<f32>
    %32 = stablehlo.multiply %31, %30 : tensor<f32>
    %33 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %34 = stablehlo.subtract %33, %arg0 : tensor<f32>
    %35 = stablehlo.constant dense<1.4142135623730951> : tensor<f32>
    %36 = stablehlo.multiply %arg1, %35 : tensor<f32>
    %37 = stablehlo.divide %34, %36 : tensor<f32>
    %38 = chlo.erf %37 : tensor<f32> -> tensor<f32>
    %39 = stablehlo.constant dense<1.0> : tensor<f32>
    %40 = stablehlo.add %39, %38 : tensor<f32>
    %41 = stablehlo.constant dense<0.5> : tensor<f32>
    %42 = stablehlo.multiply %41, %40 : tensor<f32>
    %43 = stablehlo.subtract %32, %42 : tensor<f32>
    %44 = stablehlo.log %43 : tensor<f32>
    %45 = stablehlo.subtract %23, %44 : tensor<f32>
    return %45 : tensor<f32>
  }
}
