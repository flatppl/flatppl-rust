module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %3 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %4 = stablehlo.divide %3, %arg1 : tensor<f32>
    %5 = stablehlo.atan2 %4, %0 : tensor<f32>
    %6 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %7 = stablehlo.multiply %6, %5 : tensor<f32>
    %8 = stablehlo.constant dense<0.5> : tensor<f32>
    %9 = stablehlo.add %8, %7 : tensor<f32>
    %10 = stablehlo.subtract %0, %9 : tensor<f32>
    %11 = stablehlo.constant dense<0.0> : tensor<f32>
    %12 = stablehlo.compare EQ, %10, %11 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %13 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %14 = stablehlo.select %12, %0, %10 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.log %14 : tensor<f32>
    %16 = stablehlo.select %12, %13, %15 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %17 = stablehlo.abs %16 : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %20 = stablehlo.constant dense<true> : tensor<i1>
    %21 = stablehlo.compare LE, %8, %18 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %22 = stablehlo.and %20, %21 : tensor<i1>
    %23 = stablehlo.select %22, %8, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %25 = stablehlo.log %arg1 : tensor<f32>
    %26 = stablehlo.negate %25 : tensor<f32>
    %27 = stablehlo.subtract %23, %arg0 : tensor<f32>
    %28 = stablehlo.divide %27, %arg1 : tensor<f32>
    %29 = stablehlo.multiply %28, %28 : tensor<f32>
    %30 = stablehlo.add %0, %29 : tensor<f32>
    %31 = stablehlo.log %30 : tensor<f32>
    %32 = stablehlo.negate %31 : tensor<f32>
    %33 = stablehlo.add %24, %26 : tensor<f32>
    %34 = stablehlo.add %33, %32 : tensor<f32>
    %35 = stablehlo.negate %18 : tensor<f32>
    %36 = stablehlo.select %22, %34, %35 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %37 = stablehlo.subtract %36, %16 : tensor<f32>
    %38 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %39 = stablehlo.select %19, %37, %38 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %39 : tensor<f32>
  }
}
